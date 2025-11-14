"""
Reasoning Dataset Module

Loads and processes Chain-of-Thought reasoning datasets.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import json
import os

import torch
from torch.utils.data import Dataset, DataLoader

try:
    from datasets import load_dataset  # type: ignore
except Exception:
    load_dataset = None  # noqa: N816

try:
    from transformers import AutoTokenizer  # type: ignore
except Exception:
    AutoTokenizer = None  # type: ignore


class CoTDataset(Dataset):
    """
    Dataset for Chain-of-Thought reasoning problems.
    
    Format: problem + rationale + answer
    """
    
    def __init__(
        self,
        problems: List[str],
        answers: List[str],
        rationales: List[str],
        tokenizer,
        seq_len: int = 512,
    ):
        """
        Initialize CoT dataset.
        
        Args:
            problems: List of problem strings
            answers: List of answer strings
            rationales: List of CoT rationale strings
            tokenizer: Tokenizer for encoding
            seq_len: Maximum sequence length
        """
        assert len(problems) == len(answers) == len(rationales), \
            "Problems, answers, and rationales must have same length"
        
        self.problems = problems
        self.answers = answers
        self.rationales = rationales
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        # Format examples
        self.samples: List[Dict[str, torch.Tensor]] = []
        self._prepare_samples()
        
    def _prepare_samples(self):
        """Prepare tokenized samples."""
        for problem, rationale, answer in zip(self.problems, self.rationales, self.answers):
            # Format: <problem>{problem}</problem>\n<think>{rationale}</think>\n<answer>{answer}</answer>
            text = f"<problem>{problem}</problem>\n<think>{rationale}</think>\n<answer>{answer}</answer>"
            
            # Encode
            if hasattr(self.tokenizer, "encode"):
                if hasattr(self.tokenizer, "__call__"):
                    encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
                    token_ids = encoded["input_ids"][0].tolist()
                else:
                    token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            else:
                # Fallback: character-level
                if hasattr(self.tokenizer, "stoi"):
                    token_ids = [self.tokenizer.stoi.get(ch, 0) for ch in text]
                else:
                    raise ValueError("Tokenizer must support encoding")
            
            # Truncate/pad to seq_len
            if len(token_ids) > self.seq_len:
                token_ids = token_ids[:self.seq_len]
            else:
                token_ids = token_ids + [0] * (self.seq_len - len(token_ids))
            
            # Create input and labels (shifted by 1 for next-token prediction)
            input_ids = token_ids[:-1]
            labels = token_ids[1:]
            
            self.samples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


class ReasoningDataLoader:
    """Load reasoning datasets from various sources."""
    
    @staticmethod
    def load_bespoke_stratos_17k(
        tokenizer,
        seq_len: int = 512,
        split: str = "train",
    ) -> CoTDataset:
        """
        Load Bespoke-Stratos-17k dataset.
        
        Args:
            tokenizer: Tokenizer for encoding
            seq_len: Maximum sequence length
            split: Dataset split ("train" or "validation")
            
        Returns:
            CoTDataset instance
        """
        if load_dataset is None:
            raise RuntimeError("datasets library not installed. Install with: pip install datasets")
        
        try:
            ds = load_dataset("bespokelabs/Bespoke-Stratos-17k", split=split)
        except Exception as e:
            raise RuntimeError(f"Failed to load Bespoke-Stratos-17k: {e}")
        
        problems = []
        answers = []
        rationales = []
        
        for ex in ds:
            # Extract problem, answer, rationale
            # Format may vary, try common fields
            problem = ex.get("problem", ex.get("question", ex.get("input", "")))
            answer = ex.get("answer", ex.get("output", ""))
            rationale = ex.get("rationale", ex.get("reasoning", ex.get("chain_of_thought", "")))
            
            if problem and answer:
                problems.append(str(problem))
                answers.append(str(answer))
                rationales.append(str(rationale) if rationale else "")
        
        return CoTDataset(problems, answers, rationales, tokenizer, seq_len)
    
    @staticmethod
    def load_open_thoughts_114k(
        tokenizer,
        seq_len: int = 512,
        split: str = "train",
    ) -> CoTDataset:
        """
        Load OpenThoughts-114k dataset.
        
        Args:
            tokenizer: Tokenizer for encoding
            seq_len: Maximum sequence length
            split: Dataset split ("train" or "validation")
            
        Returns:
            CoTDataset instance
        """
        if load_dataset is None:
            raise RuntimeError("datasets library not installed. Install with: pip install datasets")
        
        try:
            ds = load_dataset("open-thoughts/OpenThoughts-114k", split=split)
        except Exception as e:
            raise RuntimeError(f"Failed to load OpenThoughts-114k: {e}")
        
        problems = []
        answers = []
        rationales = []
        
        for ex in ds:
            problem = ex.get("problem", ex.get("question", ex.get("input", "")))
            answer = ex.get("answer", ex.get("output", ""))
            rationale = ex.get("rationale", ex.get("reasoning", ex.get("chain_of_thought", "")))
            
            if problem and answer:
                problems.append(str(problem))
                answers.append(str(answer))
                rationales.append(str(rationale) if rationale else "")
        
        return CoTDataset(problems, answers, rationales, tokenizer, seq_len)
    
    @staticmethod
    def load_from_jsonl(
        jsonl_path: str,
        tokenizer,
        seq_len: int = 512,
    ) -> CoTDataset:
        """
        Load CoT dataset from JSONL file.
        
        Expected format (one JSON object per line):
        {"problem": "...", "answer": "...", "rationale": "..."}
        
        Args:
            jsonl_path: Path to JSONL file
            tokenizer: Tokenizer for encoding
            seq_len: Maximum sequence length
            
        Returns:
            CoTDataset instance
        """
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
        
        problems = []
        answers = []
        rationales = []
        
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                    problem = ex.get("problem", ex.get("question", ""))
                    answer = ex.get("answer", ex.get("output", ""))
                    rationale = ex.get("rationale", ex.get("reasoning", ex.get("chain_of_thought", "")))
                    
                    if problem and answer:
                        problems.append(str(problem))
                        answers.append(str(answer))
                        rationales.append(str(rationale) if rationale else "")
                except json.JSONDecodeError:
                    continue
        
        return CoTDataset(problems, answers, rationales, tokenizer, seq_len)
    
    @staticmethod
    def create_dataloader(
        dataset: CoTDataset,
        batch_size: int = 8,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> DataLoader:
        """
        Create DataLoader from CoTDataset.
        
        Args:
            dataset: CoTDataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle
            num_workers: Number of worker processes
            
        Returns:
            DataLoader instance
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

