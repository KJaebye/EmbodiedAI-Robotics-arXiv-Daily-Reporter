# BeamDojo: Learning Agile Humanoid Locomotion on Sparse Footholds 

**Title (ZH)**: BeamDojo：在稀疏支撑点上学习灵活的人形运动 

**Authors**: Huayi Wang, Zirui Wang, Junli Ren, Qingwei Ben, Tao Huang, Weinan Zhang, Jiangmiao Pang  

**Link**: [PDF](https://arxiv.org/pdf/2502.10363)  

**Abstract**: Traversing risky terrains with sparse footholds poses a significant challenge for humanoid robots, requiring precise foot placements and stable locomotion. Existing approaches designed for quadrupedal robots often fail to generalize to humanoid robots due to differences in foot geometry and unstable morphology, while learning-based approaches for humanoid locomotion still face great challenges on complex terrains due to sparse foothold reward signals and inefficient learning processes. To address these challenges, we introduce BeamDojo, a reinforcement learning (RL) framework designed for enabling agile humanoid locomotion on sparse footholds. BeamDojo begins by introducing a sampling-based foothold reward tailored for polygonal feet, along with a double critic to balancing the learning process between dense locomotion rewards and sparse foothold rewards. To encourage sufficient trail-and-error exploration, BeamDojo incorporates a two-stage RL approach: the first stage relaxes the terrain dynamics by training the humanoid on flat terrain while providing it with task terrain perceptive observations, and the second stage fine-tunes the policy on the actual task terrain. Moreover, we implement a onboard LiDAR-based elevation map to enable real-world deployment. Extensive simulation and real-world experiments demonstrate that BeamDojo achieves efficient learning in simulation and enables agile locomotion with precise foot placement on sparse footholds in the real world, maintaining a high success rate even under significant external disturbances. 

**Abstract (ZH)**: 使用稀疏 foothold 的通用人形机器人强化步行框架 

---
