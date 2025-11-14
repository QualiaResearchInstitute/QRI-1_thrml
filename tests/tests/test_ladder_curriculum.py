import torch

from modules.ladder_curriculum import LadderCurriculumConfig, LadderCurriculumGenerator


def test_ladder_curriculum_generator_returns_expected_levels():
    cfg = LadderCurriculumConfig(
        levels=3,
        min_fraction=0.25,
        max_fraction=1.0,
        random_truncate=False,
        noise_prob=0.0,
        include_original=True,
        seed=123,
    )
    gen = LadderCurriculumGenerator(seq_len=12, config=cfg)

    batch_size = 2
    input_ids = torch.arange(12).repeat(batch_size, 1)
    labels = input_ids.clone()

    levels = gen.generate(input_ids, labels)
    assert len(levels) == 3
    fractions = [lvl.difficulty for lvl in levels]
    assert fractions == sorted(fractions)
    assert fractions[-1] == 1.0
    assert levels[-1].length == 12


def test_ladder_curriculum_noise_mask_is_shared_between_inputs_and_labels():
    cfg = LadderCurriculumConfig(
        levels=1,
        min_fraction=0.5,
        max_fraction=0.5,
        random_truncate=False,
        noise_prob=0.3,
        include_original=False,
        seed=7,
    )
    gen = LadderCurriculumGenerator(seq_len=10, config=cfg, pad_token_id=0)
    input_ids = torch.ones(1, 10, dtype=torch.long)
    labels = torch.ones(1, 10, dtype=torch.long) * 2

    level = gen.generate(input_ids, labels)[0]
    orig_inputs = input_ids[:, : level.length]
    orig_labels = labels[:, : level.length]

    input_noise_mask = level.input_ids != orig_inputs
    label_noise_mask = level.labels != orig_labels
    assert torch.equal(input_noise_mask, label_noise_mask)

