import torch

from matrixpfn.generator.base import GeneratorConfig, MatrixDomain
from matrixpfn.generator.registry import build_training_registry, MatrixGeneratorRegistry
from matrixpfn.generator.online import OnlineMatrixDataset


def build_dataset(
    domains: list[str],
    grid_sizes: tuple[int, ...],
    device: torch.device,
    num_context_pairs: int = 1,
) -> tuple[OnlineMatrixDataset, int]:
    domain_enums = [MatrixDomain(d.strip()) for d in domains]
    domain_weights = {d: 1.0 / len(domain_enums) for d in domain_enums}

    config = GeneratorConfig(grid_sizes=grid_sizes)
    full_registry = build_training_registry(config, device)

    missing_domains = [d for d in domain_enums if d not in full_registry.generators]
    if missing_domains:
        missing_names = [d.value for d in missing_domains]
        available_names = [d.value for d in full_registry.generators]
        raise ValueError(
            f"Domains not found in registry: {missing_names}. "
            f"Available: {available_names}"
        )

    selected_generators = {
        domain: full_registry.generators[domain]
        for domain in domain_enums
    }
    registry = MatrixGeneratorRegistry(selected_generators)

    print(f"Training domains ({len(selected_generators)}):")
    for domain in domain_enums:
        print(f"  {domain.value}: {domain_weights[domain]:.0%}")

    dataset = OnlineMatrixDataset(registry, num_context_pairs, domain_weights=domain_weights)
    return dataset, len(selected_generators)
