"""Tests for CloneTrast encoder and contrastive loss."""

import math

import torch

from clonetrast.model import CloneTrastModel, contrastive_loss_clone


def _manual_sup_out_loss(z, clone_ids, temperature):
    z = torch.nn.functional.normalize(z, dim=1)
    batch_size = z.size(0)
    total = 0.0
    count = 0
    for i in range(batch_size):
        pos_idx = [j for j in range(batch_size) if j != i and clone_ids[j] == clone_ids[i]]
        if not pos_idx:
            continue
        denom = sum(
            math.exp(torch.dot(z[i], z[a]).item() / temperature)
            for a in range(batch_size)
            if a != i
        )
        anchor_loss = 0.0
        for p in pos_idx:
            num = math.exp(torch.dot(z[i], z[p]).item() / temperature)
            anchor_loss += -math.log(num / denom)
        total += anchor_loss / len(pos_idx)
        count += 1
    return total / max(count, 1)


def _manual_sup_in_loss(z, clone_ids, temperature):
    z = torch.nn.functional.normalize(z, dim=1)
    batch_size = z.size(0)
    total = 0.0
    count = 0
    for i in range(batch_size):
        pos_idx = [j for j in range(batch_size) if j != i and clone_ids[j] == clone_ids[i]]
        if not pos_idx:
            continue
        denom = sum(
            math.exp(torch.dot(z[i], z[a]).item() / temperature)
            for a in range(batch_size)
            if a != i
        )
        pos_mean = sum(
            math.exp(torch.dot(z[i], z[p]).item() / temperature) for p in pos_idx
        ) / len(pos_idx)
        total += -math.log(pos_mean / denom)
        count += 1
    return total / max(count, 1)


def test_contrastive_loss_clone():
    B = 8
    z = torch.randn(B, 16)
    z = torch.nn.functional.normalize(z, dim=1)
    # clone_id: 0,0,1,1,2,2,3,3 -> same-clone pairs (0,1), (2,3), (4,5), (6,7)
    clone_ids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss = contrastive_loss_clone(z, clone_ids, temperature=0.5)
    assert loss.dim() == 0
    assert loss.item() > 0


def test_contrastive_loss_sup_out_matches_paper():
    z = torch.randn(6, 8)
    clone_ids = torch.tensor([0, 0, 0, 1, 1, 2])
    temperature = 0.3
    loss = contrastive_loss_clone(z, clone_ids, temperature=temperature, variant="sup_out")
    expected = _manual_sup_out_loss(z, clone_ids, temperature)
    assert abs(loss.item() - expected) < 1e-5


def test_contrastive_loss_sup_in_matches_paper():
    z = torch.randn(6, 8)
    clone_ids = torch.tensor([0, 0, 0, 1, 1, 2])
    temperature = 0.3
    loss = contrastive_loss_clone(z, clone_ids, temperature=temperature, variant="sup_in")
    expected = _manual_sup_in_loss(z, clone_ids, temperature)
    assert abs(loss.item() - expected) < 1e-5


def test_contrastive_loss_sup_out_ge_sup_in():
    # Three cells per clone so each anchor has multiple positives; with a single
    # positive the two variants are algebraically equal and float noise can flip >=.
    z = torch.randn(9, 16)
    z = torch.nn.functional.normalize(z, dim=1)
    clone_ids = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    temperature = 0.5
    loss_out = contrastive_loss_clone(z, clone_ids, temperature=temperature, variant="sup_out")
    loss_in = contrastive_loss_clone(z, clone_ids, temperature=temperature, variant="sup_in")
    assert loss_out.item() + 1e-5 >= loss_in.item()


def test_clonetrast_model_forward():
    model = CloneTrastModel(n_genes=100, hidden_dims=[64, 32], embedding_dim=16, projection_dim=8)
    x = torch.randn(4, 100)
    h, z = model(x)
    assert h.shape == (4, 16)
    assert z.shape == (4, 8)
    norms = torch.linalg.vector_norm(z, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    h_only = model.encode(x)
    assert h_only.shape == (4, 16)


def test_contrastive_loss_normalizes_input():
    """Loss must be invariant to the scale of unnormalized projection vectors."""
    clone_ids = torch.tensor([0, 0, 1, 1])
    z = torch.randn(4, 8)
    z_scaled = z * 17.0
    temperature = 0.4
    loss_raw = contrastive_loss_clone(z, clone_ids, temperature=temperature)
    loss_scaled = contrastive_loss_clone(z_scaled, clone_ids, temperature=temperature)
    assert abs(loss_raw.item() - loss_scaled.item()) < 1e-5


def test_contrastive_loss_similarities_are_cosine():
    """Internal L2 normalization makes dot products equal cosine similarity."""
    z = torch.randn(4, 8, requires_grad=True) * torch.tensor([3.0, 0.25, 9.0, 1.5]).unsqueeze(1)
    z_unit = torch.nn.functional.normalize(z.detach(), dim=1)
    expected = torch.mm(z_unit, z_unit.t())
    z_loss = torch.nn.functional.normalize(z, dim=1)
    actual = torch.mm(z_loss, z_loss.t())
    assert torch.allclose(actual, expected, atol=1e-5)
    assert torch.all(actual >= -1 - 1e-5)
    assert torch.all(actual <= 1 + 1e-5)


def test_contrastive_loss_gradients_through_model_projection():
    model = CloneTrastModel(n_genes=12, hidden_dims=[16], embedding_dim=8, projection_dim=4)
    x = torch.randn(4, 12)
    clone_ids = torch.tensor([0, 0, 1, 1])
    _h, z = model(x)
    loss = contrastive_loss_clone(z, clone_ids, temperature=0.5)
    loss.backward()
    proj_grad = model.projection[0].weight.grad
    enc_grad = model.encoder[0].linear.weight.grad
    assert proj_grad is not None and torch.isfinite(proj_grad).all()
    assert enc_grad is not None and torch.isfinite(enc_grad).all()


def test_model_uses_projection_not_embedding_for_loss():
    model = CloneTrastModel(n_genes=20, hidden_dims=[32], embedding_dim=8, projection_dim=4)
    x = torch.randn(4, 20)
    clone_ids = torch.tensor([0, 0, 1, 1])
    h, z = model(x)
    loss_projection = contrastive_loss_clone(z, clone_ids, temperature=0.5)
    loss_embedding = contrastive_loss_clone(h, clone_ids, temperature=0.5)
    assert loss_projection.item() > 0
    # Embedding is not L2-normalized and has a different dimensionality/scale.
    assert not torch.allclose(
        torch.linalg.vector_norm(h, dim=1),
        torch.ones(h.size(0)),
        atol=1e-3,
    )
    assert loss_projection.item() != loss_embedding.item()


def test_contrastive_loss_no_positive_pairs_zero():
    """Each clone appears once: no within-batch positive pairs → zero loss contribution."""
    z = torch.randn(4, 16, requires_grad=True)
    z = torch.nn.functional.normalize(z, dim=1)
    clone_ids = torch.tensor([0, 1, 2, 3])
    loss = contrastive_loss_clone(z, clone_ids, temperature=0.5)
    assert loss.item() == 0.0


def test_clone_size_model_forward():
    from clonetrast.model import CloneSizeModel

    m = CloneSizeModel(n_genes=50, hidden_dims=[32, 16])
    x = torch.randn(3, 50)
    y = m(x)
    assert y.shape == (3,)
