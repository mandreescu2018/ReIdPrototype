import torch

class PatchMemory:

    def __init__(self, momentum=0.1, num=1):

        self.name = []
        self.agent = []
        self.momentum = momentum
        self.num = num
        
        self.camid = []
        self.pid = []
        


    def get_soft_label(self, path, feat_list, pid=None, camid=None):

        feat = torch.stack(feat_list, dim=0)

        feat = feat[:, ::self.num, :]
        

        position = []


        # update the agent
        for j,p in enumerate(path):

            current_soft_feat = feat[:, j, :].detach()
            if current_soft_feat.is_cuda:
                current_soft_feat = current_soft_feat.cpu()
            key  = p
            if key not in self.name:
                self.name.append(key)
                self.camid.append(camid[j])
                self.pid.append(pid[j])
                self.agent.append(current_soft_feat)
                ind = self.name.index(key)
                position.append(ind)
                
            else:
                ind = self.name.index(key)
                tmp = self.agent.pop(ind)
                tmp = tmp*(1-self.momentum) + self.momentum*current_soft_feat
                self.agent.insert(ind, tmp)
                position.append(ind)

        if len(position) != 0:
            position = torch.tensor(position).cuda()
    
        agent = torch.stack(self.agent, dim=1).cuda()
        return agent, position
    
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output