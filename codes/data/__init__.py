'''create dataset and dataloader'''
import logging
import torch
import torch.utils.data
# from light import light, light_init

#@light_init(params={"training_framework": "pytorch_ddp", "enable_optimizations": True, "application_scenario": "Torch-CV"})
def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    phase = dataset_opt['phase']
    if phase == 'train':
        if opt['dist']:
            world_size = torch.distributed.get_world_size()
            num_workers = dataset_opt['n_workers']
            assert dataset_opt['batch_size'] % world_size == 0
            batch_size = dataset_opt['batch_size'] // world_size
            shuffle = False
        else:
            num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
            batch_size = dataset_opt['batch_size']
            shuffle = True
        # return light.io.get_data_loader(dataset, num_replicas=light.cc.size(), rank=light.cc.rank(), batch_size=batch_size, shuffle=shuffle, sampler=sampler, batch_sampler=None, num_workers=num_workers,
        # pin_memory=True, drop_last=True, timeout=0,
        # worker_init_fn=None, multiprocessing_context=None, generator=None)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=True)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                                           pin_memory=True)

# def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
#     phase = dataset_opt['phase']
#     if phase == 'train':
#         if opt['dist']:
#             world_size = torch.distributed.get_world_size()
#             num_workers = dataset_opt['n_workers']
#             assert dataset_opt['batch_size'] % world_size == 0
#             batch_size = dataset_opt['batch_size'] // world_size
#             shuffle = False
#         else:
#             num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
#             batch_size = dataset_opt['batch_size']
#             shuffle = True
#         return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
#                                            num_workers=num_workers, sampler=sampler, drop_last=True,
#                                            pin_memory=True)
#     else:
#         return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
#                                            pin_memory=True)


def create_dataset(dataset_opt):
    mode = dataset_opt['mode']
    if mode == 'LQGTRN':
        from data.LQGTRN_dataset import LQGTRNDataset as D
    elif mode == 'LQGTSN':
        from data.LQGTSN_dataset import LQGTSNDataset as D
    elif mode == 'GTLQx': # load generated HR-LR image pairs, and replace with x4
        from data.GTLQx_dataset import GTLQxDataset as D
        # dataset = D(dataset_opt)
    elif mode == 'GTLQnpy': # load generated HR-LR image pairs
        from data.GTLQnpy_dataset import GTLQnpyDataset as D
        # dataset = D(dataset_opt)
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    dataset = D(dataset_opt)

    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
