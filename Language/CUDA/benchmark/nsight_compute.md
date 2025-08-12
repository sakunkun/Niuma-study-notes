官网操作手册：https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html

结果解读：

用ncu去观察[PPMP第五章](../PPMP/chapter-05/)实现的一个算子，直接`ncu -k  regex:matrixMul_shared_kernel_  python test.py`，结果如下：
```shell
matrixMul_shared_kernel_2(float *, float *, float *, int, int, int) (32, 128, 1)x(16, 16, 1), Context 1, Stream 7, Device 0, CC 8.9
Section: GPU Speed Of Light Throughput
# GPU 吞吐量指标，显示了GPU各个组件的利用率
----------------------- ----------- ------------
Metric Name             Metric Unit Metric Value
----------------------- ----------- ------------
DRAM Frequency                  Ghz         8.99    # GPU显存的工作频率
SM Frequency                    Ghz         1.06    # 流式多处理器(Streaming Multiprocessor)的工作频率
Elapsed Cycles                cycle      757,149    # 内核执行的总时钟周期数
Memory Throughput                 %        93.66    # 内存总体吞吐量利用率，非常高，说明内存访问很频繁
DRAM Throughput                   %         1.71    # 显存吞吐量利用率，很低说明主要使用的是缓存
Duration                         us       711.01    # 内核执行时间，约0.7毫秒
L1/TEX Cache Throughput           %        94.61    # L1缓存/纹理缓存吞吐量，非常高
L2 Cache Throughput               %        17.71    # L2缓存吞吐量
SM Active Cycles              cycle   750,757.01    # 流式多处理器(Streaming Multiprocessor)的活跃时钟周期数
Compute (SM) Throughput           %        93.66    # 计算单元吞吐量，非常高
----------------------- ----------- ------------

INF   This workload is utilizing greater than 80.0% of the available compute or memory performance of the device.   
        To further improve performance, work will likely need to be shifted from the most utilized to another unit.   
        Start by analyzing workloads in the Compute Workload Analysis section.                                        

Section: Launch Statistics
# 启动统计，描述了内核启动的配置参数
-------------------------------- --------------- ---------------
Metric Name                          Metric Unit    Metric Value
-------------------------------- --------------- ---------------
Block Size                                                   256    # 每个线程块的线程数 (16×16=256)
Function Cache Configuration                     CachePreferNone
Grid Size                                                  4,096    # 网格大小，表示为线程块的数量
Registers Per Thread             register/thread              36    # 每个线程的寄存器数量
Shared Memory Configuration Size           Kbyte           65.54    # 共享内存大小
Driver Shared Memory Per Block       Kbyte/block            1.02    # 每个线程块的驱动共享内存大小
Dynamic Shared Memory Per Block       byte/block               0
Static Shared Memory Per Block       Kbyte/block            2.05
# SMs                                         SM             142    # 流式多处理器(Streaming Multiprocessor)的数量
Stack Size                                                 1,024    # 栈大小
Threads                                   thread       1,048,576    # 线程总数
# TPCs                                                        71    # 线程束(warp)的数量
Enabled TPC IDs                                              all
Uses Green Context                                             0
Waves Per SM                                                4.81    # 平均每个SM执行的波数
-------------------------------- --------------- ---------------
# Waves Per SM的计算方式 总线程块数/SM数/每个SM能承载的最大线程块数
# 这里就是4096/142/min(24, 6, 21, 6) = 4.81
# 下面的优化意见也给出了，该内核执行时，有4个满波，1个部分波，部分波只有689个线程块，意味着162x6-689=163个块槽位空闲

OPT   Est. Speedup: 20%                                                                                             
        A wave of thread blocks is defined as the maximum number of blocks that can be executed in parallel on the    
        target GPU. The number of blocks in a wave depends on the number of multiprocessors and the theoretical       
        occupancy of the kernel. This kernel launch results in 4 full waves and a partial wave of 689 thread blocks.  
        Under the assumption of a uniform execution duration of all thread blocks, this partial wave may account for  
        up to 20.0% of the total runtime of this kernel. Try launching a grid with no partial wave. The overall       
        impact of this tail effect also lessens with the number of full waves executed for a grid. See the Hardware   
        Model (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-model) description for     
        more details on launch configurations.                                                                        

Section: Occupancy
# 占用率，显示了SM资源的限制因素
------------------------------- ----------- ------------
Metric Name                     Metric Unit Metric Value
------------------------------- ----------- ------------
Block Limit SM                        block           24    # 每个SM的最大线程块数
Block Limit Registers                 block            6    # 寄存器限制的最大块数（这是当前内核的瓶颈）
Block Limit Shared Mem                block           21    # 共享内存限制的最大块数
Block Limit Warps                     block            6    # 线程束限制的最大块数
Theoretical Active Warps per SM        warp           48    # 理论上的活跃线程束数
Theoretical Occupancy                     %          100    # 理论上的占用率
Achieved Occupancy                        %        92.15    # 实际的占用率，非常接近理论值，说明SM资源利用率很高
Achieved Active Warps Per SM           warp        44.23    # 实际的活跃线程束数
------------------------------- ----------- ------------

Section: GPU and Memory Workload Distribution
# GPU和内存工作负载分布，显示了各个组件的活跃周期
-------------------------- ----------- ------------
Metric Name                Metric Unit Metric Value
-------------------------- ----------- ------------
Average DRAM Active Cycles       cycle   109,269.33    # 平均DRAM活跃周期数
Total DRAM Elapsed Cycles        cycle   76,709,888    # DRAM总活跃周期数
Average L1 Active Cycles         cycle   750,757.01    # 平均L1活跃周期数
Total L1 Elapsed Cycles          cycle  107,689,608    # L1总活跃周期数
Average L2 Active Cycles         cycle   953,335.10    # 平均L2活跃周期数
Total L2 Elapsed Cycles          cycle   47,363,328    # L2总活跃周期数
Average SM Active Cycles         cycle   750,757.01    # 平均SM活跃周期数
Total SM Elapsed Cycles          cycle  107,689,608    # SM总活跃周期数
Average SMSP Active Cycles       cycle   750,732.51    # 平均SMSP活跃周期数
Total SMSP Elapsed Cycles        cycle  430,758,432    # SMSP总活跃周期数
-------------------------- ----------- ------------
```