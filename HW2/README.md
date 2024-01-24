
Remark:
  In each training step of maml, though the tasks in the batch are processed sequentially, they do not interfere with each other. For each task, we acquire self._meta_parameters, which is not updated until all tasks in the batch are processed. 
