import torch
import cv2


def apply_path(
    img: torch.Tensor,
    x: int,
    y: int,
    size: int,
) -> None:
    cv2.circle(img, (x, y), size, [255], -1)


def single_vector_element_cost_f(
    prediction,
    ground_truth,
    element_index: int,
) -> float:
    assert 0 <= element_index <= 3
    assert len(ground_truth.shape) == 1
    res = ((prediction[element_index] - ground_truth[element_index]) ** 2).detach().numpy().squeeze()
    return res


def optimize_for_one_image(
    img: torch.Tensor,
    ground_truth: torch.Tensor,
    size: int,
    x: int,
    y: int,
    model,
    cost_f,
) -> float:
    x = int(x)
    y = int(y)
    img_copy = img.clone()
    apply_path(
        img=img_copy[0].numpy(),
        x=x,
        y=y,
        size=size,
    )
    # - as we want to mimizing the cost
    cost = -cost_f(
        prediction=model(img_copy.unsqueeze(0)),
        ground_truth=ground_truth,
    )
    return cost
    

def optimize_f(x, dataloader):
    assert len(x) == 2
    # try with fixed image instead of whole dataset
    img_tensor, ground_truth = next(iter(dataloader))
    return optimize_for_one_image(
        img=img_tensor,
        ground_truth=ground_truth,
        size=10,
        x=x[0],
        y=x[1]
    )
    