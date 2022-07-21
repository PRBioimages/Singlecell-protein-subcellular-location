from utils import *
import tqdm
from configs import Config
import torch

try:
    from apex import amp
except:
    pass


def Pseudo_train(cfg: Config, model, train_dl, _, loss_func, optimizer, save_path, scheduler, writer, tune=None):
    print(f'[ ! ] pos weight: {cfg.loss.pos_weight}')
    print(f'[ ! ] cell weight: {cfg.loss.cellweight}')
    print(f'[ ! ] img weight: {cfg.loss.imgweight}')
    pos_weight = torch.ones(19).cuda() * cfg.loss.pos_weight
    print(f'[ √ ] {cfg.data.celllabel} training')
    try:
        optimizer.zero_grad()
        for epoch in range(cfg.train.num_epochs):
            model.train()
            if not tune:
                tq = tqdm.tqdm(train_dl)
            else:
                tq = train_dl
            basic_lr = optimizer.param_groups[0]['lr']
            losses = []
            imageloss = []
            Calibrationloss = []
            # training by amp
            scaler = torch.cuda.amp.GradScaler()
            for i, (ipt, CELL_label, IMAGE_label) in enumerate(tq):
                ipt = ipt.view(-1, ipt.shape[-3], ipt.shape[-2], ipt.shape[-1])
                CELL_label = CELL_label.view(-1, CELL_label.shape[-1])

                IMAGE_label = IMAGE_label.cuda()
                ipt, CELL_label = ipt.cuda(), CELL_label.cuda()

                if cfg.scheduler.warm_up and epoch == 0:
                    # warm up
                    length = len(train_dl)
                    initial_lr = basic_lr / length
                    optimizer.param_groups[0]['lr'] = initial_lr * (i + 1)

                with torch.cuda.amp.autocast():
                    cell, image = model(ipt, cfg.experiment.count)
                    # loss = loss_func(output, lbl)
                    loss_cell = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')(cell, CELL_label)
                    # loss_cell2 = mae()(torch.sigmoid(cell), lbl)
                    # loss_attentionmap = torch.nn.MSELoss()(matrix_calibration, AttentionMap)

                    loss_exp = loss_func(image, IMAGE_label)
                    if not len(loss_cell.shape) == 0:
                        loss_cell = loss_cell.mean()
                    if not len(loss_exp.shape) == 0:
                        loss_exp = loss_exp.mean()
                    loss = cfg.loss.cellweight * loss_cell + cfg.loss.imgweight * loss_exp
                    losses.append(loss.item())
                    imageloss.append(loss_exp.item())
                    Calibrationloss.append(loss_cell.item())

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
                    info = {'Iloss':np.array(imageloss).mean(),
                                   'Closs':np.array(Calibrationloss).mean(), 'lr':optimizer.param_groups[0]['lr']}
                    tq.set_postfix(info)

            if len(cfg.basic.GPU)>1:
                torch.save(model.module.state_dict(), save_path / 'checkpoints/f{}_epoch-{}.pth'.format(
                    cfg.experiment.run_fold, epoch))
            else:
                torch.save(model.state_dict(), save_path / 'checkpoints/f{}_epoch-{}.pth'.format(
                    cfg.experiment.run_fold, epoch))


            print(('[ √ ] epochs: {}, train loss: {:.4f}'
                   ).format(
                epoch, np.array(losses).mean()))
            writer.add_scalar('train_f{}/loss'.format(cfg.experiment.run_fold), np.mean(losses), epoch)
            writer.add_scalar('train_f{}/cell_loss'.format(cfg.experiment.run_fold), np.mean(Calibrationloss), epoch)
            writer.add_scalar('train_f{}/img_loss'.format(cfg.experiment.run_fold), np.mean(imageloss), epoch)
            writer.add_scalar('train_f{}/lr'.format(cfg.experiment.run_fold), optimizer.param_groups[0]['lr'], epoch)
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
