import os

import torch
from tqdm import tqdm

from nets.unet_training import CE_Loss, Dice_loss, Focal_Loss
from utils.utils import get_lr
from utils.utils_metrics import f_score


def fit_one_epoch(
    model_train,
    model,
    loss_history,
    eval_callback,
    optimizer,
    epoch,
    epoch_step,
    epoch_step_val,
    gen,
    gen_val,
    Epoch,
    cuda,
    dice_loss,
    focal_loss,
    cls_weights,
    num_classes,
    fp16,
    scaler,
    save_period,
    save_dir,
    local_rank=0,
):
    """
    Train and validate the model for one epoch.
    执行一个 epoch 的训练与验证。

    Notes / 说明:
    1. Images have already been letterboxed and normalized in dataloader.py.
       图像已在 dataloader.py 中完成 letterbox 与归一化预处理。
    2. No additional preprocessing is required here before forward.
       因此前向传播前不需要额外预处理。
    3. Debug information is printed in the first two iterations of epoch 0
       to help diagnose the 'all-background prediction' issue.
       为了排查“预测全背景”问题，在第 0 个 epoch 的前两步打印前景像素调试信息。
    """
    train_loss_total = 0.0
    train_fscore_total = 0.0

    val_loss_total = 0.0
    val_fscore_total = 0.0

    if local_rank == 0:
        print("Start Train")
        train_pbar = tqdm(
            total=epoch_step,
            desc=f"Epoch {epoch + 1}/{Epoch}",
            postfix=dict,
            mininterval=0.3,
        )

    model_train.train()

    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        images, target_masks, target_onehot = batch
        # images: [B, 3, H, W]
        # target_masks: [B, H, W]
        # target_onehot: [B, H, W, num_classes + 1]

        with torch.no_grad():
            class_weights = torch.from_numpy(cls_weights)
            if cuda:
                images = images.cuda(local_rank, non_blocking=True)
                target_masks = target_masks.cuda(local_rank, non_blocking=True)
                target_onehot = target_onehot.cuda(local_rank, non_blocking=True)
                class_weights = class_weights.cuda(local_rank, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # --------------------------------------------------------- #
        # Forward pass and loss computation / 前向传播与损失计算
        # --------------------------------------------------------- #
        if not fp16:
            outputs = model_train(images)

            if focal_loss:
                loss = Focal_Loss(outputs, target_masks, class_weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, target_masks, class_weights, num_classes=num_classes)

            if dice_loss:
                dice_term = Dice_loss(outputs, target_onehot)
                loss = loss + dice_term

            with torch.no_grad():
                batch_fscore = f_score(outputs, target_onehot)

            loss.backward()
            optimizer.step()
        else:
            from torch.amp import autocast

            with autocast("cuda"):
                outputs = model_train(images)

                if focal_loss:
                    loss = Focal_Loss(outputs, target_masks, class_weights, num_classes=num_classes)
                else:
                    loss = CE_Loss(outputs, target_masks, class_weights, num_classes=num_classes)

                if dice_loss:
                    dice_term = Dice_loss(outputs, target_onehot)
                    loss = loss + dice_term

                with torch.no_grad():
                    batch_fscore = f_score(outputs, target_onehot)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # --------------------------------------------------------- #
        # Debug foreground statistics / 调试前景像素统计
        # --------------------------------------------------------- #
        if local_rank == 0 and epoch == 0 and iteration < 2:
            with torch.no_grad():
                pred_argmax = torch.argmax(outputs, dim=1)
                pred_foreground_pixels = (pred_argmax != 0).sum().item()
                gt_foreground_pixels = (target_masks != 0).sum().item()
                print(
                    f"[Train-DEBUG] step {iteration + 1}: "
                    f"pred_fore={pred_foreground_pixels}, gt_fore={gt_foreground_pixels}"
                )

        train_loss_total += loss.item()
        train_fscore_total += batch_fscore.item()

        if local_rank == 0:
            train_pbar.set_postfix(
                **{
                    "total_loss": train_loss_total / (iteration + 1),
                    "f_score": train_fscore_total / (iteration + 1),
                    "lr": get_lr(optimizer),
                }
            )
            train_pbar.update(1)

    if local_rank == 0:
        train_pbar.close()
        print("Finish Train")
        print("Start Validation")
        val_pbar = tqdm(
            total=epoch_step_val,
            desc=f"Epoch {epoch + 1}/{Epoch}",
            postfix=dict,
            mininterval=0.3,
        )

    # ------------------------------------------------------------- #
    # Validation stage / 验证阶段
    # ------------------------------------------------------------- #
    model_train.eval()

    with torch.no_grad():
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break

            images, target_masks, target_onehot = batch
            class_weights = torch.from_numpy(cls_weights)

            if cuda:
                images = images.cuda(local_rank, non_blocking=True)
                target_masks = target_masks.cuda(local_rank, non_blocking=True)
                target_onehot = target_onehot.cuda(local_rank, non_blocking=True)
                class_weights = class_weights.cuda(local_rank, non_blocking=True)

            outputs = model_train(images)

            if focal_loss:
                loss = Focal_Loss(outputs, target_masks, class_weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, target_masks, class_weights, num_classes=num_classes)

            if dice_loss:
                dice_term = Dice_loss(outputs, target_onehot)
                loss = loss + dice_term

            batch_fscore = f_score(outputs, target_onehot)

            # ----------------------------------------------------- #
            # Validation debug in the first two iterations
            # 验证阶段前两步打印调试信息
            # ----------------------------------------------------- #
            if local_rank == 0 and epoch == 0 and iteration < 2:
                pred_argmax = torch.argmax(outputs, dim=1)
                pred_foreground_pixels = (pred_argmax != 0).sum().item()
                gt_foreground_pixels = (target_masks != 0).sum().item()
                print(
                    f"[Val-DEBUG] step {iteration + 1}: "
                    f"pred_fore={pred_foreground_pixels}, gt_fore={gt_foreground_pixels}"
                )

            val_loss_total += loss.item()
            val_fscore_total += batch_fscore.item()

            if local_rank == 0:
                val_pbar.set_postfix(
                    **{
                        "val_loss": val_loss_total / (iteration + 1),
                        "f_score": val_fscore_total / (iteration + 1),
                        "lr": get_lr(optimizer),
                    }
                )
                val_pbar.update(1)

    if local_rank == 0:
        val_pbar.close()
        print("Finish Validation")

        average_train_loss = train_loss_total / epoch_step
        average_val_loss = val_loss_total / epoch_step_val

        # --------------------------------------------------------- #
        # Record training history / 记录训练历史
        # --------------------------------------------------------- #
        loss_history.append_loss(epoch + 1, average_train_loss, average_val_loss)

        # --------------------------------------------------------- #
        # Instance-level evaluation callback / 实例级评估回调
        # --------------------------------------------------------- #
        eval_callback.on_epoch_end(epoch + 1, model_train)

        print(f"Epoch:{epoch + 1}/{Epoch}")
        print("Total Loss: %.3f || Val Loss: %.3f " % (average_train_loss, average_val_loss))

        # --------------------------------------------------------- #
        # Save checkpoints / 保存权重文件
        # --------------------------------------------------------- #
        if (epoch + 1) % save_period == 0 or (epoch + 1) == Epoch:
            periodic_save_path = os.path.join(
                save_dir,
                "ep%03d-loss%.3f-val_loss%.3f.pth"
                % (epoch + 1, average_train_loss, average_val_loss),
            )
            torch.save(model.state_dict(), periodic_save_path)

        if len(loss_history.val_loss) <= 1 or average_val_loss <= min(loss_history.val_loss):
            print("Save best model to best_epoch_weights.pth")
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))


def fit_one_epoch_no_val(
    model_train,
    model,
    loss_history,
    optimizer,
    epoch,
    epoch_step,
    gen,
    Epoch,
    cuda,
    dice_loss,
    focal_loss,
    cls_weights,
    num_classes,
    fp16,
    scaler,
    save_period,
    save_dir,
    local_rank=0,
):
    """
    Train the model for one epoch without a validation set.
    无验证集版本的单个 epoch 训练函数。

    The function signature is kept unchanged for compatibility.
    为保证兼容性，函数接口保持不变。
    """
    train_loss_total = 0.0
    train_fscore_total = 0.0

    if local_rank == 0:
        print("Start Train")
        train_pbar = tqdm(
            total=epoch_step,
            desc=f"Epoch {epoch + 1}/{Epoch}",
            postfix=dict,
            mininterval=0.3,
        )

    model_train.train()

    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        images, target_masks, target_onehot = batch

        with torch.no_grad():
            class_weights = torch.from_numpy(cls_weights)
            if cuda:
                images = images.cuda(local_rank, non_blocking=True)
                target_masks = target_masks.cuda(local_rank, non_blocking=True)
                target_onehot = target_onehot.cuda(local_rank, non_blocking=True)
                class_weights = class_weights.cuda(local_rank, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if not fp16:
            outputs = model_train(images)

            if focal_loss:
                loss = Focal_Loss(outputs, target_masks, class_weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, target_masks, class_weights, num_classes=num_classes)

            if dice_loss:
                dice_term = Dice_loss(outputs, target_onehot)
                loss = loss + dice_term

            with torch.no_grad():
                batch_fscore = f_score(outputs, target_onehot)

            loss.backward()
            optimizer.step()
        else:
            from torch.amp import autocast

            with autocast("cuda"):
                outputs = model_train(images)

                if focal_loss:
                    loss = Focal_Loss(outputs, target_masks, class_weights, num_classes=num_classes)
                else:
                    loss = CE_Loss(outputs, target_masks, class_weights, num_classes=num_classes)

                if dice_loss:
                    dice_term = Dice_loss(outputs, target_onehot)
                    loss = loss + dice_term

                with torch.no_grad():
                    batch_fscore = f_score(outputs, target_onehot)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # --------------------------------------------------------- #
        # Debug foreground statistics / 调试前景像素统计
        # --------------------------------------------------------- #
        if local_rank == 0 and epoch == 0 and iteration < 2:
            with torch.no_grad():
                pred_argmax = torch.argmax(outputs, dim=1)
                pred_foreground_pixels = (pred_argmax != 0).sum().item()
                gt_foreground_pixels = (target_masks != 0).sum().item()
                print(
                    f"[Train(no-val)-DEBUG] step {iteration + 1}: "
                    f"pred_fore={pred_foreground_pixels}, gt_fore={gt_foreground_pixels}"
                )

        train_loss_total += loss.item()
        train_fscore_total += batch_fscore.item()

        if local_rank == 0:
            train_pbar.set_postfix(
                **{
                    "total_loss": train_loss_total / (iteration + 1),
                    "f_score": train_fscore_total / (iteration + 1),
                    "lr": get_lr(optimizer),
                }
            )
            train_pbar.update(1)

    if local_rank == 0:
        train_pbar.close()

        average_train_loss = train_loss_total / epoch_step
        loss_history.append_loss(epoch + 1, average_train_loss)

        print(f"Epoch:{epoch + 1}/{Epoch}")
        print("Total Loss: %.3f" % average_train_loss)

        # --------------------------------------------------------- #
        # Save checkpoints / 保存权重文件
        # --------------------------------------------------------- #
        if (epoch + 1) % save_period == 0 or (epoch + 1) == Epoch:
            periodic_save_path = os.path.join(
                save_dir,
                "ep%03d-loss%.3f.pth" % (epoch + 1, average_train_loss),
            )
            torch.save(model.state_dict(), periodic_save_path)

        if len(loss_history.losses) <= 1 or average_train_loss <= min(loss_history.losses):
            print("Save best model to best_epoch_weights.pth")
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))