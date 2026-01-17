import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, AutoModel


@dataclass
class PipelineStep:
    name: str
    prob: float = 1.0
    apply_to: str = "either"
    weight: Optional[float] = None


class PrefixPerturber:
    def __init__(self, instruction_prefixes: List[str], output_prefixes: List[str]):
        self.instruction_prefixes = instruction_prefixes
        self.output_prefixes = output_prefixes

    def apply(self, text: str, field: str) -> str:
        if field == "instruction" and self.instruction_prefixes:
            return f"{random.choice(self.instruction_prefixes)} {text}"
        if field == "output" and self.output_prefixes:
            return f"{random.choice(self.output_prefixes)} {text}"
        return text


class BAEPerturber:
    def __init__(self, device: str = "cpu", model_name: str = "bert-base-uncased"):
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.mlm = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        self.embedder = AutoModel.from_pretrained(model_name).to(device)

        self.mlm.eval()
        self.embedder.eval()

    def _sentence_embedding(self, sentence: str) -> np.ndarray:
        inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True).to(self.device)

        with torch.no_grad():
            outputs = self.embedder(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1)

        return emb.cpu().numpy()

    def _get_candidate_words(self, masked_sentence: str, max_candidates: int) -> List[str]:
        inputs = self.tokenizer(masked_sentence, return_tensors="pt").to(self.device)

        mask_token_index = torch.where(
            inputs["input_ids"] == self.tokenizer.mask_token_id
        )[1]

        with torch.no_grad():
            logits = self.mlm(**inputs).logits

        top_k = torch.topk(
            logits[0, mask_token_index, :],
            max_candidates,
            dim=-1,
        ).indices[0].tolist()

        return [self.tokenizer.decode([idx]).strip() for idx in top_k]

    def perturb_sentence(
        self,
        sentence: str,
        stopwords: set,
        similarity_threshold: float,
        max_candidates: int,
    ) -> str:
        words = sentence.split()

        candidate_indices = [
            i
            for i, w in enumerate(words)
            if w.lower().strip(".,!?;:") not in stopwords
            and w.strip(".,!?;:").isalpha()
        ]

        if not candidate_indices:
            return sentence

        random.shuffle(candidate_indices)
        original_emb = self._sentence_embedding(sentence)

        for idx in candidate_indices:
            masked_words = words.copy()
            masked_words[idx] = self.tokenizer.mask_token
            masked_sentence = " ".join(masked_words)

            candidates = self._get_candidate_words(masked_sentence, max_candidates)

            best_sent = sentence
            best_sim = -1.0

            for cand in candidates:
                if cand.lower() == words[idx].lower():
                    continue
                if not cand.isalpha():
                    continue

                new_words = words.copy()
                new_words[idx] = cand
                new_sentence = " ".join(new_words)

                new_emb = self._sentence_embedding(new_sentence)
                sim = cosine_similarity(original_emb, new_emb)[0][0]

                if sim > best_sim:
                    best_sim = sim
                    best_sent = new_sentence

            if best_sim >= similarity_threshold:
                return best_sent

        return sentence


class Seq2SickPerturber:
    def __init__(self, model_name: str = "t5-small", device: str = "cpu"):
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        self.embedding = self.model.get_input_embeddings().weight

    def _sentence_embedding(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)

        with torch.no_grad():
            emb = self.model.get_input_embeddings()(inputs["input_ids"])
            emb = emb.mean(dim=1)

        return emb.cpu().numpy()

    def perturb_sentence(
        self,
        src_text: str,
        tgt_text: str,
        max_perturbations: int,
        topk_neighbors: int,
        topk_tokens: int,
        similarity_threshold: float,
        accept_prob: float,
    ) -> str:
        inputs = self.tokenizer(src_text, return_tensors="pt", truncation=True).to(self.device)

        input_ids = inputs["input_ids"].clone().detach()

        embeddings = self.model.get_input_embeddings()(input_ids)
        embeddings.retain_grad()

        labels = self.tokenizer(tgt_text, return_tensors="pt", truncation=True)["input_ids"].to(
            self.device
        )

        outputs = self.model(inputs_embeds=embeddings, labels=labels)
        outputs.loss.backward()

        grad_norms = embeddings.grad.norm(dim=-1)[0]
        sorted_indices = torch.argsort(grad_norms, descending=True).tolist()

        token_indices = random.sample(
            sorted_indices[:topk_tokens],
            k=min(topk_tokens, max_perturbations),
        )

        original_emb = self._sentence_embedding(src_text)

        words = input_ids[0].tolist()
        perturbed_ids = words.copy()
        num_changes = 0

        for idx in token_indices:
            if num_changes >= max_perturbations:
                break

            original_id = words[idx]
            original_vec = self.embedding[original_id].unsqueeze(0)

            sims = cosine_similarity(
                original_vec.cpu().detach().numpy(),
                self.embedding.cpu().detach().numpy(),
            )[0]

            nearest = np.argsort(-sims)[1:topk_neighbors].tolist()

            random.shuffle(nearest)

            for new_id in nearest:
                if new_id == original_id:
                    continue

                candidate_ids = perturbed_ids.copy()
                candidate_ids[idx] = int(new_id)

                candidate_text = self.tokenizer.decode(candidate_ids, skip_special_tokens=True)

                new_emb = self._sentence_embedding(candidate_text)
                sim = cosine_similarity(original_emb, new_emb)[0][0]

                if sim >= similarity_threshold:
                    if random.random() < accept_prob:
                        perturbed_ids[idx] = int(new_id)
                        num_changes += 1
                        break

        return self.tokenizer.decode(perturbed_ids, skip_special_tokens=True)


def load_prefixes(
    prefix_json: Optional[str],
    output_prefixes_json: Optional[str],
    hf_dataset: str,
    hf_split: str,
    hf_field: str,
) -> Dict[str, List[str]]:
    instruction_prefixes: List[str] = []
    output_prefixes: List[str] = []

    if prefix_json:
        with open(prefix_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        instruction_prefixes = payload.get("instruction", [])
        output_prefixes = payload.get("output", [])
    else:
        ds = load_dataset(hf_dataset)
        if hf_split not in ds:
            raise ValueError(f"Split not found: {hf_split}")
        instruction_prefixes = ds[hf_split][hf_field]

    if output_prefixes_json:
        output_prefixes = json.loads(output_prefixes_json)

    return {
        "instruction": instruction_prefixes,
        "output": output_prefixes,
    }


def select_fields(apply_to: str, entry: Dict) -> List[str]:
    has_instruction = bool(entry.get("instruction"))
    has_output = bool(entry.get("output"))

    if apply_to == "instruction":
        return ["instruction"] if has_instruction else []
    if apply_to == "output":
        return ["output"] if has_output else []
    if apply_to == "both":
        fields = []
        if has_instruction:
            fields.append("instruction")
        if has_output:
            fields.append("output")
        return fields

    if has_instruction and has_output:
        return [random.choice(["instruction", "output"])]
    if has_instruction:
        return ["instruction"]
    if has_output:
        return ["output"]
    return []


def weighted_choice(steps: List[PipelineStep]) -> PipelineStep:
    weights = [s.weight if s.weight is not None else s.prob for s in steps]
    total = sum(weights) if weights else 0.0
    if total <= 0:
        return steps[0]
    r = random.random() * total
    upto = 0.0
    for step, w in zip(steps, weights):
        upto += w
        if upto >= r:
            return step
    return steps[-1]


class PerturbationPipeline:
    def __init__(
        self,
        steps: List[PipelineStep],
        mode: str,
        prefix_config: Dict[str, List[str]],
        bae_model: str,
        seq2sick_model: str,
        device: str,
        bae_stopwords: set,
        bae_similarity_threshold: float,
        bae_max_candidates: int,
        seq2sick_max_perturbations: int,
        seq2sick_topk_neighbors: int,
        seq2sick_topk_tokens: int,
        seq2sick_similarity_threshold: float,
        seq2sick_accept_prob: float,
    ):
        self.steps = steps
        self.mode = mode
        self.prefix_config = prefix_config
        self.bae_model = bae_model
        self.seq2sick_model = seq2sick_model
        self.device = device
        self.bae_stopwords = bae_stopwords
        self.bae_similarity_threshold = bae_similarity_threshold
        self.bae_max_candidates = bae_max_candidates
        self.seq2sick_max_perturbations = seq2sick_max_perturbations
        self.seq2sick_topk_neighbors = seq2sick_topk_neighbors
        self.seq2sick_topk_tokens = seq2sick_topk_tokens
        self.seq2sick_similarity_threshold = seq2sick_similarity_threshold
        self.seq2sick_accept_prob = seq2sick_accept_prob
        self._perturbers: Dict[str, object] = {}

    def _get_perturber(self, name: str):
        if name in self._perturbers:
            return self._perturbers[name]

        if name == "prefix":
            perturber = PrefixPerturber(
                instruction_prefixes=self.prefix_config.get("instruction", []),
                output_prefixes=self.prefix_config.get("output", []),
            )
        elif name == "bae":
            perturber = BAEPerturber(device=self.device, model_name=self.bae_model)
        elif name == "seq2sick":
            perturber = Seq2SickPerturber(model_name=self.seq2sick_model, device=self.device)
        else:
            raise ValueError(f"Unknown step: {name}")

        self._perturbers[name] = perturber
        return perturber

    def _apply_step(self, step: PipelineStep, entry: Dict, field: str) -> str:
        text = entry.get(field, "")
        if not text:
            return text

        perturber = self._get_perturber(step.name)

        if step.name == "prefix":
            return perturber.apply(text, field)

        if step.name == "bae":
            return perturber.perturb_sentence(
                text,
                stopwords=self.bae_stopwords,
                similarity_threshold=self.bae_similarity_threshold,
                max_candidates=self.bae_max_candidates,
            )

        if step.name == "seq2sick":
            if field == "instruction":
                tgt_text = entry.get("output") or text
            else:
                tgt_text = text
            return perturber.perturb_sentence(
                text,
                tgt_text,
                max_perturbations=self.seq2sick_max_perturbations,
                topk_neighbors=self.seq2sick_topk_neighbors,
                topk_tokens=self.seq2sick_topk_tokens,
                similarity_threshold=self.seq2sick_similarity_threshold,
                accept_prob=self.seq2sick_accept_prob,
            )

        return text

    def apply_entry(self, entry: Dict) -> Dict:
        new_entry = entry.copy()

        if self.mode == "sample":
            step = weighted_choice(self.steps)
            fields = select_fields(step.apply_to, new_entry)
            if random.random() <= step.prob:
                for field in fields:
                    new_entry[field] = self._apply_step(step, new_entry, field)
            return new_entry

        for step in self.steps:
            fields = select_fields(step.apply_to, new_entry)
            if random.random() > step.prob:
                continue
            for field in fields:
                new_entry[field] = self._apply_step(step, new_entry, field)

        return new_entry


def parse_pipeline(pipeline_arg: Optional[str]) -> List[PipelineStep]:
    if pipeline_arg is None:
        return [
            PipelineStep(name="prefix", prob=0.5, apply_to="output"),
            PipelineStep(name="bae", prob=0.5, apply_to="instruction"),
            PipelineStep(name="seq2sick", prob=0.5, apply_to="either"),
        ]

    if os.path.isfile(pipeline_arg):
        with open(pipeline_arg, "r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = json.loads(pipeline_arg)

    steps = []
    for item in payload:
        steps.append(
            PipelineStep(
                name=item["name"],
                prob=float(item.get("prob", 1.0)),
                apply_to=item.get("apply_to", "either"),
                weight=item.get("weight"),
            )
        )
    return steps


def load_input(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of objects")
    return data


def save_output(path: str, data: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run_cli(
    default_pipeline: Optional[List[Dict]] = None,
    default_output: Optional[str] = None,
    default_input: Optional[str] = None,
) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        default=default_input or "data/simple_train.json",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=default_output or "perturbed.json",
    )
    parser.add_argument("--pipeline", type=str)
    parser.add_argument("--pipeline_mode", type=str, choices=["sequential", "sample"], default="sequential")
    parser.add_argument("--seed", type=int)

    parser.add_argument("--prefix_json", type=str)
    parser.add_argument("--output_prefixes", type=str)
    parser.add_argument("--hf_dataset", type=str, default="MultiverseComputingCAI/llm-refusal-evaluation")
    parser.add_argument("--hf_split", type=str, default="jailbreakbench")
    parser.add_argument("--hf_field", type=str, default="prompt")

    parser.add_argument("--bae_model", type=str, default="bert-base-uncased")
    parser.add_argument("--seq2sick_model", type=str, default="t5-small")

    parser.add_argument("--bae_similarity_threshold", type=float, default=0.6)
    parser.add_argument("--bae_max_candidates", type=int, default=10)

    parser.add_argument("--seq2sick_max_perturbations", type=int, default=1)
    parser.add_argument("--seq2sick_topk_neighbors", type=int, default=50)
    parser.add_argument("--seq2sick_topk_tokens", type=int, default=5)
    parser.add_argument("--seq2sick_similarity_threshold", type=float, default=0.7)
    parser.add_argument("--seq2sick_accept_prob", type=float, default=0.7)

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    pipeline_arg = args.pipeline
    if pipeline_arg is None and default_pipeline is not None:
        pipeline_arg = json.dumps(default_pipeline)

    steps = parse_pipeline(pipeline_arg)

    prefix_config = load_prefixes(
        prefix_json=args.prefix_json,
        output_prefixes_json=args.output_prefixes,
        hf_dataset=args.hf_dataset,
        hf_split=args.hf_split,
        hf_field=args.hf_field,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "in",
        "on",
        "at",
        "and",
        "or",
        "of",
        "to",
        "with",
        "for",
        "by",
        "when",
        "what",
        "do",
        "does",
        "did",
    }

    pipeline = PerturbationPipeline(
        steps=steps,
        mode=args.pipeline_mode,
        prefix_config=prefix_config,
        bae_model=args.bae_model,
        seq2sick_model=args.seq2sick_model,
        device=device,
        bae_stopwords=stopwords,
        bae_similarity_threshold=args.bae_similarity_threshold,
        bae_max_candidates=args.bae_max_candidates,
        seq2sick_max_perturbations=args.seq2sick_max_perturbations,
        seq2sick_topk_neighbors=args.seq2sick_topk_neighbors,
        seq2sick_topk_tokens=args.seq2sick_topk_tokens,
        seq2sick_similarity_threshold=args.seq2sick_similarity_threshold,
        seq2sick_accept_prob=args.seq2sick_accept_prob,
    )

    data = load_input(args.input_file)
    out = [pipeline.apply_entry(item) for item in data]
    save_output(args.output_file, out)

    print(f"Saved perturbed data to {args.output_file}")


if __name__ == "__main__":
    run_cli()
