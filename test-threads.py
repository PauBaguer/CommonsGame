import torch
import timeit
import matplotlib.pyplot as plt
runtimes = []
threads = [t for t in range(10, 33, 2)]
for t in threads:
    torch.set_num_threads(t)
    r = timeit.timeit(setup = "import torch; x = torch.randn(1024, 1024); y = torch.randn(1024, 1024)", stmt="torch.mm(x, y)", number=1000)
    runtimes.append(r)


plt.plot(threads, runtimes)
plt.savefig("test-threads.png")