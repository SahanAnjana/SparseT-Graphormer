# --------------------------------------------------------
# References:
# mae_st: https://github.com/facebookresearch/mae_st
# --------------------------------------------------------

import torch
import utils.misc as misc
from utils.meters import PredTestMeter


@torch.no_grad()
def test(data_loader, model, device, args, fp32=False):
    task = args.task

    criterion = torch.nn.MSELoss()
    metric_logger = PredTestMeter(delimiter="  ")

    header = "Test:"

    # switch to evaluation atlas
    model.eval()

    for batched_data in metric_logger.log_every(data_loader, 10, header):
        batched_data = misc.prepare_batch(batched_data, device=device)
        scaler = None if 'scaler' not in batched_data else batched_data['scaler']

        samples, targets, target_shape = misc.get_samples_targets(batched_data, task)

        # compute output
        with torch.cuda.amp.autocast(enabled=False):
            preds = model(batched_data)
            if scaler:
                outputs = scaler.inverse_transform(preds, args)
            else:
                outputs = preds

        batch_size = samples.shape[0]
        metric_logger.store_predictions(outputs.detach(), targets.detach())

        metrics = misc.forecasting_acc(
            outputs,
            targets,
            target_shape=None if args.n_hist == args.n_pred or args.n_pred > args.n_hist
            else target_shape
        )

        metric_logger.meters['mae'].update(metrics['MAE'].item(), n=batch_size)
        metric_logger.meters['rmse'].update(metrics['RMSE'].item(), n=batch_size)
        metric_logger.meters['mape'].update(metrics['MAPE'].item(), n=batch_size)

    metrics = metric_logger.finalize_metrics(
        target_shape=None if args.n_hist == args.n_pred or args.n_pred > args.n_hist
        else target_shape
    )
    print(
        f"*****************************************************************************\n"
        f"MAE of the network on {len(data_loader)} points: {metrics['MAE']:.4f}\n"
        f"MAPE of the network on {len(data_loader)} points: {metrics['MAPE']:.7f}\n"
        f"RMSE of the network on {len(data_loader)} points: {metrics['RMSE']:.4f}\n"
    )
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
