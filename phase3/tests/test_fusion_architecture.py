import torch

from phase3.fusion.network import FUSION_LABELS, GatedFusionNetwork


def test_gated_fusion_accepts_sparse_text_and_emits_three_heads():
    model = GatedFusionNetwork(image_dim=512, text_dim=20, hidden_dim=16)
    image = torch.randn(2, 512)
    indices = torch.tensor([[0, 0, 1, 1], [1, 7, 3, 12]])
    values = torch.tensor([0.4, 1.0, 0.6, 0.8])
    text = torch.sparse_coo_tensor(indices, values, size=(2, 20)).coalesce()
    output = model(image, text)
    assert output["fusion_logits"].shape == (2, len(FUSION_LABELS))
    assert output["image_logits"].shape == (2, 2)
    assert output["text_logits"].shape == (2, 2)
    assert output["image_embedding"].shape == output["text_embedding"].shape == (2, 16)
    loss = output["fusion_logits"].sum() + output["image_logits"].sum() + output["text_logits"].sum()
    loss.backward()
    assert model.text_projection[0].weight.grad is not None
    assert model.image_projection[0].weight.grad is not None
