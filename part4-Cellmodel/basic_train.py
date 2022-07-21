from utils import *
import tqdm
from configs import Config
import torch

try:
    from apex import amp
except:
    pass


def basic_train(cfg: Config, model, train_dl, valid_dl, loss_func, optimizer, save_path, scheduler, writer, tune=None):
    print('[ √ ] Cell-based model training')
    try:
        optimizer.zero_grad()
        for epoch in range(cfg.train.num_epochs):
            if scheduler and cfg.scheduler.name in ['StepLR']:
                scheduler.step(epoch)
            model.train()
            if not tune:
                tq = tqdm.tqdm(train_dl)
            else:
                tq = train_dl
            basic_lr = optimizer.param_groups[0]['lr']
            losses = []

            scaler = torch.cuda.amp.GradScaler()
            for i, (imgs, label) in enumerate(tq):
                cell_label = label.cuda()
                imgs = imgs.cuda()

                if cfg.scheduler.warm_up and epoch == 0:
                    # warm up
                    length = len(train_dl)
                    initial_lr = basic_lr / length
                    optimizer.param_groups[0]['lr'] = initial_lr * (i + 1)

                with torch.cuda.amp.autocast():
                    output = model(imgs)
                    loss_cell = loss_func(output, cell_label)

                    if cfg.loss.weight_value:
                        calss_weight = torch.Tensor(cfg.loss.weight_value)
                        calss_weight = calss_weight / calss_weight.sum()
                        loss_cell = torch.mean(loss_cell * calss_weight.cuda())
                    else:
                        if not len(loss_cell.shape) == 0:
                            loss_cell = loss_cell.mean()
                    loss = loss_cell
                    losses.append(loss.item())

                scaler.scale(loss).backward()

                if cfg.train.clip:
                    scaler.unscale_(optimizer)
                    # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if cfg.scheduler.name in ['CyclicLR', 'OneCycleLR', 'CosineAnnealingLR']:
                    if epoch == 0 and cfg.scheduler.warm_up:
                        pass
                    else:
                        scheduler.step()
                if not tune:
                    tq.set_postfix(loss=np.array(losses).mean(), lr=optimizer.param_groups[0]['lr'])

            if len(cfg.basic.GPU) > 1:
                torch.save(model.module.state_dict(), save_path / 'checkpoints/f{}_epoch-{}.pth'.format(
                    cfg.experiment.run_fold, epoch))
            else:
                torch.save(model.state_dict(), save_path / 'checkpoints/f{}_epoch-{}.pth'.format(
                    cfg.experiment.run_fold, epoch))

            # validate_loss, accuracy, auc = basic_validate(model, valid_dl, loss_func, cfg, tune)

            print(('[ √ ] epochs: {}, train loss: {:.4f}, '
                   ).format(
                epoch, np.array(losses).mean()))
            writer.add_scalar('train_f{}/loss'.format(cfg.experiment.run_fold), np.mean(losses), epoch)
            writer.add_scalar('train_f{}/lr'.format(cfg.experiment.run_fold), optimizer.param_groups[0]['lr'], epoch)
            # writer.add_scalar('valid_f{}/loss'.format(cfg.experiment.run_fold), validate_loss, epoch)

            with open(save_path / 'train.log', 'a') as fp:
                fp.write('{}\t{:.8f}\t{:.4f}\n'.format(
                    epoch, optimizer.param_groups[0]['lr'], np.array(losses).mean()))

    except KeyboardInterrupt:
        print('[ X ] Ctrl + c, QUIT')
        if len(cfg.basic.GPU) > 1:
            torch.save(model.module.state_dict(), save_path / 'checkpoints/quit{}_epoch-{}.pth'.format(
                cfg.experiment.run_fold, epoch))
        else:
            torch.save(model.state_dict(), save_path / 'checkpoints/quit{}_epoch-{}.pth'.format(
                cfg.experiment.run_fold, epoch))


def basic_validate(mdl, dl, loss_func, cfg, tune=None):
    mdl.eval()
    with torch.no_grad():
        results = []
        losses, predicted, predicted_p, truth = [], [], [], []
        for i, (imgs, exp_label) in enumerate(dl):
            imgs, exp_label = imgs.cuda(), exp_label.cuda()
            if cfg.basic.amp == 'Native':
                with torch.cuda.amp.autocast():
                    output = mdl(imgs)
                    loss = loss_func(output, exp_label)
                    if not len(loss.shape) == 0:
                        loss = loss.mean()
                    output = output.float()
            else:
                output = mdl(imgs)
                loss = loss_func(output, exp_label)
                if not len(loss.shape) == 0:
                    loss = loss.mean()
            losses.append(loss.item())
            predicted.append(torch.sigmoid(output.cpu()).numpy())
            truth.append(exp_label.cpu().numpy())
            results.append({
                'step': i,
                'loss': loss.item(),
            })
        predicted = np.concatenate(predicted)
        truth = np.concatenate(truth)
        val_loss = np.array(losses).mean()
        accuracy = ((predicted > 0.5) == truth).sum().astype(np.float) / truth.shape[0] / truth.shape[1]
        # auc = macro_multilabel_auc(truth, predicted, gpu=0)

        return val_loss, accuracy, 0
