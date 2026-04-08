def extract_loss_metrics(history, evaluation_result):
    metrics = {
        "train_loss": float(history.history["loss"][-1]),
    }

    if "val_loss" in history.history:
        metrics["val_loss"] = float(history.history["val_loss"][-1])

    if isinstance(evaluation_result, (list, tuple)):
        metrics["test_loss"] = float(evaluation_result[0])
        if len(evaluation_result) > 1:
            metrics["test_mae"] = float(evaluation_result[1])
    else:
        metrics["test_loss"] = float(evaluation_result)

    return metrics
