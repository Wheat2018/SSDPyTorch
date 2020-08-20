import torch


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
        # self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            # self.next_input, \
            # self.next_box_target, self.next_cls_target, \
            # self.next_ctr_target, self.next_cor_target = next(self.loader)
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            # self.next_box_target = None
            # self.next_cls_target = None
            # self.next_ctr_target = None
            # self.next_cor_target = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            for i in range(0, len(self.next_target)):
                self.next_target[i] = self.next_target[i].cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target



    def __iter__(self):
        return self


class data_prefetcher_ctr():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
        # self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            # self.next_input, \
            # self.next_box_target, self.next_cls_target, \
            # self.next_ctr_target, self.next_cor_target = next(self.loader)
            self.next_input, self.next_target, self.next_ctr_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            # self.next_box_target = None
            # self.next_cls_target = None
            self.next_ctr_target = None
            # self.next_cor_target = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_ctr_target = self.next_ctr_target.cuda(non_blocking=True)
            for i in range(0, len(self.next_target)):
                self.next_target[i] = self.next_target[i].cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        ctr_target = self.next_ctr_target
        self.preload()
        return input, target, ctr_target



    def __iter__(self):
        return self