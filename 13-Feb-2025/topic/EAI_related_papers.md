# A Real-to-Sim-to-Real Approach to Robotic Manipulation with VLM-Generated Iterative Keypoint Rewards 

**Title (ZH)**: 一种基于VLM生成的迭代关键点奖励的从真实到模拟再到真实的机器人操作方法 

**Authors**: Shivansh Patel, Xinchen Yin, Wenlong Huang, Shubham Garg, Hooshang Nayyeri, Li Fei-Fei, Svetlana Lazebnik, Yunzhu Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.08643)  

**Abstract**: Task specification for robotic manipulation in open-world environments is challenging, requiring flexible and adaptive objectives that align with human intentions and can evolve through iterative feedback. We introduce Iterative Keypoint Reward (IKER), a visually grounded, Python-based reward function that serves as a dynamic task specification. Our framework leverages VLMs to generate and refine these reward functions for multi-step manipulation tasks. Given RGB-D observations and free-form language instructions, we sample keypoints in the scene and generate a reward function conditioned on these keypoints. IKER operates on the spatial relationships between keypoints, leveraging commonsense priors about the desired behaviors, and enabling precise SE(3) control. We reconstruct real-world scenes in simulation and use the generated rewards to train reinforcement learning (RL) policies, which are then deployed into the real world-forming a real-to-sim-to-real loop. Our approach demonstrates notable capabilities across diverse scenarios, including both prehensile and non-prehensile tasks, showcasing multi-step task execution, spontaneous error recovery, and on-the-fly strategy adjustments. The results highlight IKER's effectiveness in enabling robots to perform multi-step tasks in dynamic environments through iterative reward shaping. 

**Abstract (ZH)**: 机器人 manipulation 在开放世界环境中的任务规范具有挑战性，要求具备灵活性和适应性，并能与人类意图对齐且通过迭代反馈进行演化。我们提出了一种基于视觉导向的 Python 基准奖励函数 Iterative Keypoint Reward (IKER)，作为动态任务规范。我们的框架利用视觉语言模型 (VLMs) 为多步操作任务生成和优化这些奖励函数。给定 RGB-D 观测和自由形式的语言指令，我们从场景中采样关键点并生成条件化奖励函数。IKER 利用关于预期行为的常识先验知识，基于关键点之间的空间关系，实现精确的 SE(3) 控制。我们在模拟中重建现实世界场景，并使用生成的奖励训练强化学习 (RL) 策略，然后将其部署到现实世界中，形成从现实到模拟再到现实的闭环。我们的方法在多种场景中展示了显著能力，包括各种拾取式和非拾取式任务，展示多步任务执行、自发错误恢复和实时策略调整。结果表明，IKER 在动态环境中通过迭代奖励塑形使机器人能够执行多步任务方面非常有效。 

---
# Robot Data Curation with Mutual Information Estimators 

**Title (ZH)**: 基于互信息估计器的机器人数据整理 

**Authors**: Joey Hejna, Suvir Mirchandani, Ashwin Balakrishna, Annie Xie, Ayzaan Wahid, Jonathan Tompson, Pannag Sanketi, Dhruv Shah, Coline Devin, Dorsa Sadigh  

**Link**: [PDF](https://arxiv.org/pdf/2502.08623)  

**Abstract**: The performance of imitation learning policies often hinges on the datasets with which they are trained. Consequently, investment in data collection for robotics has grown across both industrial and academic labs. However, despite the marked increase in the quantity of demonstrations collected, little work has sought to assess the quality of said data despite mounting evidence of its importance in other areas such as vision and language. In this work, we take a critical step towards addressing the data quality in robotics. Given a dataset of demonstrations, we aim to estimate the relative quality of individual demonstrations in terms of both state diversity and action predictability. To do so, we estimate the average contribution of a trajectory towards the mutual information between states and actions in the entire dataset, which precisely captures both the entropy of the state distribution and the state-conditioned entropy of actions. Though commonly used mutual information estimators require vast amounts of data often beyond the scale available in robotics, we introduce a novel technique based on k-nearest neighbor estimates of mutual information on top of simple VAE embeddings of states and actions. Empirically, we demonstrate that our approach is able to partition demonstration datasets by quality according to human expert scores across a diverse set of benchmarks spanning simulation and real world environments. Moreover, training policies based on data filtered by our method leads to a 5-10% improvement in RoboMimic and better performance on real ALOHA and Franka setups. 

**Abstract (ZH)**: imitation学习策略的表现往往取决于其训练所用的数据集。因此，工业和学术实验室在机器人领域的数据收集投资都有所增长。尽管收集的演示数据量显著增加，但很少有研究致力于评估这些数据的质量，尽管其他领域如视觉和语言的证据表明数据质量十分重要。在本文中，我们朝着解决机器人领域的数据质量问题迈出了关键一步。给定一组演示数据，我们旨在从状态多样性和动作可预测性的角度估计单个演示的质量。为此，我们估计轨迹对整个数据集中状态和动作之间互信息的平均贡献，这准确地捕捉到了状态分布的熵和动作条件熵。尽管常用的互信息估计器通常需要大量数据，往往超过机器人领域可获得的规模，我们提出了基于k-最近邻互信息估计和简单的VAE状态-动作嵌入的新技术。通过实验证明，我们的方法可以根据人类专家评分对不同基准测试中的模拟和真实环境的演示数据集进行按质量分割。此外，使用我们方法过滤后的数据训练策略在RoboMimic上的性能提高了5-10%，在真实的ALOHA和Franka系统上表现更好。 

---
# Learning to Group and Grasp Multiple Objects 

**Title (ZH)**: 学习分组和抓取多个物体 

**Authors**: Takahiro Yonemaru, Weiwei Wan, Tatsuki Nishimura, Kensuke Harada  

**Link**: [PDF](https://arxiv.org/pdf/2502.08452)  

**Abstract**: Simultaneously grasping and transporting multiple objects can significantly enhance robotic work efficiency and has been a key research focus for decades. The primary challenge lies in determining how to push objects, group them, and execute simultaneous grasping for respective groups while considering object distribution and the hardware constraints of the robot. Traditional rule-based methods struggle to flexibly adapt to diverse scenarios. To address this challenge, this paper proposes an imitation learning-based approach. We collect a series of expert demonstrations through teleoperation and train a diffusion policy network, enabling the robot to dynamically generate action sequences for pushing, grouping, and grasping, thereby facilitating efficient multi-object grasping and transportation. We conducted experiments to evaluate the method under different training dataset sizes, varying object quantities, and real-world object scenarios. The results demonstrate that the proposed approach can effectively and adaptively generate multi-object grouping and grasping strategies. With the support of more training data, imitation learning is expected to be an effective approach for solving the multi-object grasping problem. 

**Abstract (ZH)**: 同时抓取和运输多个物体能显著提升机器人工作效率，一直是数十年来的关键研究方向。主要挑战在于如何推动物体、分组并执行针对各自组别的同时抓取，同时考虑物体分布和机器人的硬件限制。传统的基于规则的方法难以灵活适应各种场景。为应对这一挑战，本文提出了一种模拟学习方法。我们通过遥控操作收集了一系列专家演示，并训练了一个扩散策略网络，使机器人能够动态生成推动物体、分组和抓取的动作序列，从而促进高效的多物体抓取和运输。我们在不同训练数据集大小、不同物体数量和真实世界物体场景下进行了实验评估。结果表明，所提出的方法能够有效地且适应性地生成多物体分组和抓取策略。随着训练数据的支持，模拟学习有望成为解决多物体抓取问题的有效方法。 

---
# CordViP: Correspondence-based Visuomotor Policy for Dexterous Manipulation in Real-World 

**Title (ZH)**: 基于对应关系的视觉运动策略及其在真实世界中的灵巧 manipulation 

**Authors**: Yankai Fu, Qiuxuan Feng, Ning Chen, Zichen Zhou, Mengzhen Liu, Mingdong Wu, Tianxing Chen, Shanyu Rong, Jiaming Liu, Hao Dong, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.08449)  

**Abstract**: Achieving human-level dexterity in robots is a key objective in the field of robotic manipulation. Recent advancements in 3D-based imitation learning have shown promising results, providing an effective pathway to achieve this goal. However, obtaining high-quality 3D representations presents two key problems: (1) the quality of point clouds captured by a single-view camera is significantly affected by factors such as camera resolution, positioning, and occlusions caused by the dexterous hand; (2) the global point clouds lack crucial contact information and spatial correspondences, which are necessary for fine-grained dexterous manipulation tasks. To eliminate these limitations, we propose CordViP, a novel framework that constructs and learns correspondences by leveraging the robust 6D pose estimation of objects and robot proprioception. Specifically, we first introduce the interaction-aware point clouds, which establish correspondences between the object and the hand. These point clouds are then used for our pre-training policy, where we also incorporate object-centric contact maps and hand-arm coordination information, effectively capturing both spatial and temporal dynamics. Our method demonstrates exceptional dexterous manipulation capabilities with an average success rate of 90\% in four real-world tasks, surpassing other baselines by a large margin. Experimental results also highlight the superior generalization and robustness of CordViP to different objects, viewpoints, and scenarios. Code and videos are available on this https URL. 

**Abstract (ZH)**: 实现人类水平灵巧操作是机器人操作领域的关键目标。基于3D的模仿学习的最新进展取得了有前景的结果，提供了一种有效的途径来实现这一目标。然而，获取高质量的3D表示存在两个关键问题：（1）单视角相机捕获的点云质量受相机分辨率、位置以及灵巧手引起的遮挡等因素显著影响；（2）全局点云缺乏关键的接触信息和空间对应关系，这些对于精细的灵巧操作任务是必要的。为了解决这些限制，我们提出了一种名为CordViP的新框架，通过利用对象和机器人自身感知的鲁棒6D姿态估计来构建和学习对应关系。具体而言，我们首先引入了交互感知点云，建立对象与手之间的对应关系。这些点云随后用于我们的预训练策略，我们在其中还结合了以对象为中心的接触图和手-臂协调信息，有效捕捉了空间和时间动态。我们的方法在四个实际任务中的平均成功率为90%，显著优于其他基线方法。实验结果还突显了CordViP在不同对象、视角和场景下的更强的泛化能力和鲁棒性。代码和视频可在以下链接获取。 

---
# Learning Humanoid Standing-up Control across Diverse Postures 

**Title (ZH)**: 学习 humanoid 站立起立控制 Across Diverse 姿态 

**Authors**: Tao Huang, Junli Ren, Huayi Wang, Zirui Wang, Qingwei Ben, Muning Wen, Xiao Chen, Jianan Li, Jiangmiao Pang  

**Link**: [PDF](https://arxiv.org/pdf/2502.08378)  

**Abstract**: Standing-up control is crucial for humanoid robots, with the potential for integration into current locomotion and loco-manipulation systems, such as fall recovery. Existing approaches are either limited to simulations that overlook hardware constraints or rely on predefined ground-specific motion trajectories, failing to enable standing up across postures in real-world scenes. To bridge this gap, we present HoST (Humanoid Standing-up Control), a reinforcement learning framework that learns standing-up control from scratch, enabling robust sim-to-real transfer across diverse postures. HoST effectively learns posture-adaptive motions by leveraging a multi-critic architecture and curriculum-based training on diverse simulated terrains. To ensure successful real-world deployment, we constrain the motion with smoothness regularization and implicit motion speed bound to alleviate oscillatory and violent motions on physical hardware, respectively. After simulation-based training, the learned control policies are directly deployed on the Unitree G1 humanoid robot. Our experimental results demonstrate that the controllers achieve smooth, stable, and robust standing-up motions across a wide range of laboratory and outdoor environments. Videos are available at this https URL. 

**Abstract (ZH)**: Humanoid Standing-up Control: A Reinforcement Learning Framework for Robust Sim-to-Real Transfer Across Diverse Postures 

---
# COMBO-Grasp: Learning Constraint-Based Manipulation for Bimanual Occluded Grasping 

**Title (ZH)**: COMBO-Grasp: 基于约束的学习双臂遮挡抓取操作 

**Authors**: Jun Yamada, Alexander L. Mitchell, Jack Collins, Ingmar Posner  

**Link**: [PDF](https://arxiv.org/pdf/2502.08054)  

**Abstract**: This paper addresses the challenge of occluded robot grasping, i.e. grasping in situations where the desired grasp poses are kinematically infeasible due to environmental constraints such as surface collisions. Traditional robot manipulation approaches struggle with the complexity of non-prehensile or bimanual strategies commonly used by humans in these circumstances. State-of-the-art reinforcement learning (RL) methods are unsuitable due to the inherent complexity of the task. In contrast, learning from demonstration requires collecting a significant number of expert demonstrations, which is often infeasible. Instead, inspired by human bimanual manipulation strategies, where two hands coordinate to stabilise and reorient objects, we focus on a bimanual robotic setup to tackle this challenge. In particular, we introduce Constraint-based Manipulation for Bimanual Occluded Grasping (COMBO-Grasp), a learning-based approach which leverages two coordinated policies: a constraint policy trained using self-supervised datasets to generate stabilising poses and a grasping policy trained using RL that reorients and grasps the target object. A key contribution lies in value function-guided policy coordination. Specifically, during RL training for the grasping policy, the constraint policy's output is refined through gradients from a jointly trained value function, improving bimanual coordination and task performance. Lastly, COMBO-Grasp employs teacher-student policy distillation to effectively deploy point cloud-based policies in real-world environments. Empirical evaluations demonstrate that COMBO-Grasp significantly improves task success rates compared to competitive baseline approaches, with successful generalisation to unseen objects in both simulated and real-world environments. 

**Abstract (ZH)**: 基于约束的双臂遮挡抓取学习方法（COMBO-Grasp） 

---
# End-to-End Predictive Planner for Autonomous Driving with Consistency Models 

**Title (ZH)**: 端到端一致性预测规划器 for 自动驾驶 

**Authors**: Anjian Li, Sangjae Bae, David Isele, Ryne Beeson, Faizan M. Tariq  

**Link**: [PDF](https://arxiv.org/pdf/2502.08033)  

**Abstract**: Trajectory prediction and planning are fundamental components for autonomous vehicles to navigate safely and efficiently in dynamic environments. Traditionally, these components have often been treated as separate modules, limiting the ability to perform interactive planning and leading to computational inefficiency in multi-agent scenarios. In this paper, we present a novel unified and data-driven framework that integrates prediction and planning with a single consistency model. Trained on real-world human driving datasets, our consistency model generates samples from high-dimensional, multimodal joint trajectory distributions of the ego and multiple surrounding agents, enabling end-to-end predictive planning. It effectively produces interactive behaviors, such as proactive nudging and yielding to ensure both safe and efficient interactions with other road users. To incorporate additional planning constraints on the ego vehicle, we propose an alternating direction method for multi-objective guidance in online guided sampling. Compared to diffusion models, our consistency model achieves better performance with fewer sampling steps, making it more suitable for real-time deployment. Experimental results on Waymo Open Motion Dataset (WOMD) demonstrate our method's superiority in trajectory quality, constraint satisfaction, and interactive behavior compared to various existing approaches. 

**Abstract (ZH)**: 自主车辆在动态环境中的轨迹预测与规划是确保安全高效导航的基础组件。传统上，这些组件常被视作独立模块，限制了交互规划的能力，并在多智能体场景中导致了计算效率低下。本文提出了一种新颖的统一且数据驱动的框架，将预测与规划整合到单一的一致性模型中。通过训练于实际人类驾驶数据集上，该一致性模型生成高维、多模态的环境ego与多个周围智能体的联合轨迹分布样本，实现了端到端的预测规划。该模型有效地生成了互动行为，如主动提示和礼让，以确保与其他道路使用者的安全和高效互动。为了在在线引导采样中对ego车辆施加额外的规划约束，我们提出了一种多目标引导的交替方向方法。与扩散模型相比，我们的一致性模型在较少的采样步骤中表现出更优的性能，更适合实时部署。实验结果在Waymo Open Motion Dataset (WOMD) 上表明，与各种现有方法相比，该方法在轨迹质量、约束满足和互动行为方面具有明显优势。 

---
# Optimal Actuator Attacks on Autonomous Vehicles Using Reinforcement Learning 

**Title (ZH)**: 基于强化学习的自主车辆最优执行器攻击 

**Authors**: Pengyu Wang, Jialu Li, Ling Shi  

**Link**: [PDF](https://arxiv.org/pdf/2502.07839)  

**Abstract**: With the increasing prevalence of autonomous vehicles (AVs), their vulnerability to various types of attacks has grown, presenting significant security challenges. In this paper, we propose a reinforcement learning (RL)-based approach for designing optimal stealthy integrity attacks on AV actuators. We also analyze the limitations of state-of-the-art RL-based secure controllers developed to counter such attacks. Through extensive simulation experiments, we demonstrate the effectiveness and efficiency of our proposed method. 

**Abstract (ZH)**: 随着自动驾驶车辆（AVs）的逐渐普及，它们受到了各种类型攻击的威胁，这提出了重大的安全挑战。本文提出了一种基于强化学习（RL）的方法来设计针对AV执行器的最优隐蔽性完整性攻击。我们还分析了用于抵御此类攻击的最先进的基于RL的安全控制器的局限性。通过广泛的仿真实验，我们展示了所提出方法的有效性和效率。 

---
# RoboBERT: An End-to-end Multimodal Robotic Manipulation Model 

**Title (ZH)**: RoboBERT：一种端到端多模态机器人操作模型 

**Authors**: Sicheng Wang, Jianhua Shan, Jianwei Zhang, Haozhang Gao, Hailiang Han, Yipeng Chen, Kang Wei, Chengkun Zhang, Kairos Wong, Jie Zhao, Lei Zhao, Bin Fang  

**Link**: [PDF](https://arxiv.org/pdf/2502.07837)  

**Abstract**: Embodied intelligence integrates multiple modalities, enabling agents to understand images, language, and actions simultaneously. However, existing models always depend on additional datasets or extensive pre-training to maximize performance improvements, consuming abundant training time and expensive hardware cost. To tackle this issue, we present RoboBERT, a novel end-to-end robotic manipulation model integrated with a unique training strategy. This model utilizes a CNN-based diffusion policy, enhancing and stabilizing the effectiveness of this model by separating training processes for different modalities. It also underscores the importance of data augmentation, verifying various techniques to significantly boost performance. Unlike models that depend on extra data or large foundation models, RoboBERT achieves a highly competitive success rate while using only language-labeled expert demonstrations and maintaining a relatively smaller model size. Specifically, RoboBERT achieves an average length of 4.52 on the CALVIN benchmark for \(ABCD \rightarrow D\) task, setting a new state-of-the-art (SOTA) record. Furthermore, when tested on a real robot, the model demonstrates superior performance, achieving a higher success rate than other methods trained with the same data. We propose that these concepts and methodologies of RoboBERT demonstrate extensive versatility and compatibility, contributing significantly to the development of lightweight multimodal robotic models. The code can be accessed on this https URL 

**Abstract (ZH)**: 具身智能融合多种模态，使代理能够同时理解图像、语言和动作。然而，现有模型总是依赖额外的数据集或 extensive 预训练来最大化性能改进，消耗大量的训练时间和昂贵的硬件成本。为解决这一问题，我们提出了 RoboBERT，这是一种新颖的一体化机器人操作模型，结合了独特的训练策略。该模型利用基于 CNN 的扩散策略，通过分离不同模态的训练过程来增强和稳定模型效果。它还强调了数据增强的重要性，验证了多种技术以显著提升性能。与依赖额外数据或大型基础模型的模型不同，RoboBERT 在仅使用语言标注的专家演示和较小的模型规模下，实现了高度竞争的准确率。具体而言，RoboBERT 在 CALVIN 基准上 \(ABCD \rightarrow D\) 任务中达到平均长度 4.52，创下了新的state-of-the-art（SOTA）记录。此外，在实际机器人上测试时，该模型表现出色，与使用相同数据训练的其他方法相比，成功率达到更高。我们提出，RoboBERT 这些概念和方法展示了广泛的灵活性和兼容性，对轻量级多模态机器人模型的发展做出了重要贡献。代码可在以下链接访问：this https URL。 

---
# Generative AI-Enhanced Cooperative MEC of UAVs and Ground Stations for Unmanned Surface Vehicles 

**Title (ZH)**: 基于生成式AI增强的无人机与地面站协作MEC技术在无人水面车辆中的应用 

**Authors**: Jiahao You, Ziye Jia, Chao Dong, Qihui Wu, Zhu Han  

**Link**: [PDF](https://arxiv.org/pdf/2502.08119)  

**Abstract**: The increasing deployment of unmanned surface vehicles (USVs) require computational support and coverage in applications such as maritime search and rescue. Unmanned aerial vehicles (UAVs) can offer low-cost, flexible aerial services, and ground stations (GSs) can provide powerful supports, which can cooperate to help the USVs in complex scenarios. However, the collaboration between UAVs and GSs for USVs faces challenges of task uncertainties, USVs trajectory uncertainties, heterogeneities, and limited computational resources. To address these issues, we propose a cooperative UAV and GS based robust multi-access edge computing framework to assist USVs in completing computational tasks. Specifically, we formulate the optimization problem of joint task offloading and UAV trajectory to minimize the total execution time, which is in the form of mixed integer nonlinear programming and NP-hard to tackle. Therefore, we propose the algorithm of generative artificial intelligence-enhanced heterogeneous agent proximal policy optimization (GAI-HAPPO). The proposed algorithm integrates GAI models to enhance the actor network ability to model complex environments and extract high-level features, thereby allowing the algorithm to predict uncertainties and adapt to dynamic conditions. Additionally, GAI stabilizes the critic network, addressing the instability of multi-agent reinforcement learning approaches. Finally, extensive simulations demonstrate that the proposed algorithm outperforms the existing benchmark methods, thus highlighting the potentials in tackling intricate, cross-domain issues in the considered scenarios. 

**Abstract (ZH)**: 无人驾驶水面车辆（USVs）部署的增加需要计算支持和覆盖，尤其是在海上搜救等应用中。无人驾驶航空车辆（UAVs）可以提供低成本、灵活的空中服务，地面站（GSs）可以提供强大的支持，可以合作帮助USVs应对复杂场景。然而，UAVs和GSs之间的协作面临着任务不确定性、USVs轨迹不确定性、异构性和有限的计算资源等挑战。为了解决这些问题，我们提出了一种基于合作UAV和GS的鲁棒多接入边缘计算框架，以协助USVs完成计算任务。具体而言，我们将联合任务卸载和UAV轨迹的优化问题形式化为混合整数非线性规划问题，并证明其NP-hard，因此提出了生成式人工智能增强异质代理近端策略优化算法（GAI-HAPPO）。该算法结合了GAI模型，增强了演员网络构建复杂环境和提取高级特征的能力，从而允许算法预测不确定性并适应动态条件。此外，GAI稳定了评论者网络，解决了多智能体强化学习方法的不稳定性问题。最后，大量的仿真表明，所提出的算法优于现有的基准方法，从而突显了在考虑场景中复杂跨域问题中的潜在应用。 

---
# Salience-Invariant Consistent Policy Learning for Generalization in Visual Reinforcement Learning 

**Title (ZH)**: 视觉强化学习中不变显著性的一致策略学习以实现泛化 

**Authors**: Sun Jingbo, Tu Songjun, Zhang Qichao, Chen Ke, Zhao Dongbin  

**Link**: [PDF](https://arxiv.org/pdf/2502.08336)  

**Abstract**: Generalizing policies to unseen scenarios remains a critical challenge in visual reinforcement learning, where agents often overfit to the specific visual observations of the training environment. In unseen environments, distracting pixels may lead agents to extract representations containing task-irrelevant information. As a result, agents may deviate from the optimal behaviors learned during training, thereby hindering visual this http URL address this issue, we propose the Salience-Invariant Consistent Policy Learning (SCPL) algorithm, an efficient framework for zero-shot generalization. Our approach introduces a novel value consistency module alongside a dynamics module to effectively capture task-relevant representations. The value consistency module, guided by saliency, ensures the agent focuses on task-relevant pixels in both original and perturbed observations, while the dynamics module uses augmented data to help the encoder capture dynamic- and reward-relevant representations. Additionally, our theoretical analysis highlights the importance of policy consistency for generalization. To strengthen this, we introduce a policy consistency module with a KL divergence constraint to maintain consistent policies across original and perturbed this http URL experiments on the DMC-GB, Robotic Manipulation, and CARLA benchmarks demonstrate that SCPL significantly outperforms state-of-the-art methods in terms of generalization. Notably, SCPL achieves average performance improvements of 14\%, 39\%, and 69\% in the challenging DMC video hard setting, the Robotic hard setting, and the CARLA benchmark, this http URL Page: this https URL. 

**Abstract (ZH)**: 视觉强化学习中将策略推广到未见过的场景仍然是一个关键挑战，其中代理往往会过度拟合训练环境中特定的视觉观察。在未见过的环境中，分散的像素可能导致代理提取包含任务无关信息的表示。结果，代理可能会偏离训练期间学习到的最优行为，从而妨碍视觉强化学习的推广。为了解决这一问题，我们提出了一种名为Salience-Invariant Consistent Policy Learning (SCPL) 的算法，这是一种高效的零样本推广框架。我们的方法引入了一种新颖的价值一致性模块和一个动力学模块，以有效地捕捉任务相关表示。价值一致性模块由显著性引导，确保代理在原始观察和扰动观察中都关注任务相关像素，而动力学模块则利用增强数据来帮助编码器捕捉动态和奖励相关表示。此外，我们的理论分析强调了策略一致性对于推广的重要性。为了加强这一点，我们引入了一个具有KL散度约束的策略一致性模块，以在原始观察和扰动观察中保持一致策略。在DMC-GB、机器人操作和CARLA基准测试上的实验表明，SCPL在推广性能方面显著优于最新方法，尤其是在具有挑战性的DMC视频硬设置、机器人硬设置和CARLA基准测试中分别实现了平均性能改进的14%、39%和69%。 

---
# Rhythmic sharing: A bio-inspired paradigm for zero-shot adaptation and learning in neural networks 

**Title (ZH)**: 节奏共享：一种受生物启发的零样本适应与学习神经网络范式 

**Authors**: Hoony Kang, Wolfgang Losert  

**Link**: [PDF](https://arxiv.org/pdf/2502.08644)  

**Abstract**: The brain can rapidly adapt to new contexts and learn from limited data, a coveted characteristic that artificial intelligence algorithms have struggled to mimic. Inspired by oscillatory rhythms of the mechanical structures of neural cells, we developed a learning paradigm that is based on oscillations in link strengths and associates learning with the coordination of these oscillations. We find that this paradigm yields rapid adaptation and learning in artificial neural networks. Link oscillations can rapidly change coordination, endowing the network with the ability to sense subtle context changes in an unsupervised manner. In other words, the network generates the missing contextual tokens required to perform as a generalist AI architecture capable of predicting dynamics in multiple contexts. Oscillations also allow the network to extrapolate dynamics to never-seen-before contexts. These capabilities make our learning paradigm a powerful starting point for novel models of learning and cognition. Furthermore, learning through link coordination is agnostic to the specifics of the neural network architecture, hence our study opens the door for introducing rapid adaptation and learning capabilities into leading AI models. 

**Abstract (ZH)**: 大脑能够快速适应新环境并从有限的数据中学习，这是人工智能算法一直难以模仿的宝贵特性。受神经细胞机械结构中的振荡节奏启发，我们开发了一种基于连接强度振荡的学习范式，并将学习与这些振荡的协调联系起来。我们发现，这种范式能够使人工神经网络快速适应和学习。连接振荡能够迅速改变协调，赋予网络在无监督情况下感知细微环境变化的能力。换句话说，网络生成了执行通用人工智能架构所需的缺失上下文令牌，能够预测多种环境中的动力学。振荡还使网络能够推断从未见过的环境中的动力学。这些能力使我们的学习范式成为新型学习和认知模型的强大起点。此外，通过连接协调进行学习对神经网络架构的具体细节不敏感，因此我们的研究为引入快速适应和学习能力到领先的人工智能模型中开启了大门。 

---
# Human-Centric Foundation Models: Perception, Generation and Agentic Modeling 

**Title (ZH)**: 以人类为中心的基础模型：感知、生成与自主建模 

**Authors**: Shixiang Tang, Yizhou Wang, Lu Chen, Yuan Wang, Sida Peng, Dan Xu, Wanli Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2502.08556)  

**Abstract**: Human understanding and generation are critical for modeling digital humans and humanoid embodiments. Recently, Human-centric Foundation Models (HcFMs) inspired by the success of generalist models, such as large language and vision models, have emerged to unify diverse human-centric tasks into a single framework, surpassing traditional task-specific approaches. In this survey, we present a comprehensive overview of HcFMs by proposing a taxonomy that categorizes current approaches into four groups: (1) Human-centric Perception Foundation Models that capture fine-grained features for multi-modal 2D and 3D understanding. (2) Human-centric AIGC Foundation Models that generate high-fidelity, diverse human-related content. (3) Unified Perception and Generation Models that integrate these capabilities to enhance both human understanding and synthesis. (4) Human-centric Agentic Foundation Models that extend beyond perception and generation to learn human-like intelligence and interactive behaviors for humanoid embodied tasks. We review state-of-the-art techniques, discuss emerging challenges and future research directions. This survey aims to serve as a roadmap for researchers and practitioners working towards more robust, versatile, and intelligent digital human and embodiments modeling. 

**Abstract (ZH)**: 以人类为中心的基础模型对于建模数字人类和类人实体至关重要。近年来，受通用模型（如大型语言和视觉模型）成功的启发，人类为中心的基础模型（HcFMs）逐渐兴起，将多种人类为中心的任务统一到一个框架中，超越了传统任务特定的方法。在本文综述中，我们通过提出一种分类法对当前的HcFMs进行分类，分为四类：（1）人类为中心的感知基础模型，捕捉多模态2D和3D的细粒度特征。（2）人类为中心的AIGC基础模型，生成高质量、多样化的与人类相关的内容。（3）统一感知与生成模型，将这些能力整合以增强人类理解和合成。（4）人类为中心的代理基础模型，超越感知和生成，学习类人类的智能和交互行为，用于类人实体任务。我们回顾了最先进的技术，讨论了新兴的挑战和未来的研究方向。本文综述旨在为致力于更稳健、多功能和智能的数字人类及其实体建模的研究人员和从业者提供路线图。 

---
# VSC-RL: Advancing Autonomous Vision-Language Agents with Variational Subgoal-Conditioned Reinforcement Learning 

**Title (ZH)**: VSC-RL: 基于变分子目标条件强化学习的自主视觉-语言代理提升 

**Authors**: Qingyuan Wu, Jianheng Liu, Jianye Hao, Jun Wang, Kun Shao  

**Link**: [PDF](https://arxiv.org/pdf/2502.07949)  

**Abstract**: State-of-the-art (SOTA) reinforcement learning (RL) methods enable the vision-language agents to learn from interactions with the environment without human supervision. However, they struggle with learning inefficiencies in tackling real-world complex sequential decision-making tasks, especially with sparse reward signals and long-horizon dependencies. To effectively address the issue, we introduce Variational Subgoal-Conditioned RL (VSC-RL), which reformulates the vision-language sequential decision-making task as a variational goal-conditioned RL problem, allowing us to leverage advanced optimization methods to enhance learning efficiency. Specifically, VSC-RL optimizes the SubGoal Evidence Lower BOund (SGC-ELBO), which consists of (a) maximizing the subgoal-conditioned return via RL and (b) minimizing the subgoal-conditioned difference with the reference policy. We theoretically demonstrate that SGC-ELBO is equivalent to the original optimization objective, ensuring improved learning efficiency without sacrificing performance guarantees. Additionally, for real-world complex decision-making tasks, VSC-RL leverages the vision-language model to autonomously decompose the goal into feasible subgoals, enabling efficient learning. Across various benchmarks, including challenging real-world mobile device control tasks, VSC-RL significantly outperforms the SOTA vision-language agents, achieving superior performance and remarkable improvement in learning efficiency. 

**Abstract (ZH)**: 基于变分子目标调节的强化学习方法（Variational Subgoal-Conditioned RL, VSC-RL）：解决视觉语言代理在现实世界复杂顺序决策任务中的学习效率问题 

---
# Some things to know about achieving artificial general intelligence 

**Title (ZH)**: 关于实现人工通用智能需要了解的一些事情 

**Authors**: Herbert Roitblat  

**Link**: [PDF](https://arxiv.org/pdf/2502.07828)  

**Abstract**: Current and foreseeable GenAI models are not capable of achieving artificial general intelligence because they are burdened with anthropogenic debt. They depend heavily on human input to provide well-structured problems, architecture, and training data. They cast every problem as a language pattern learning problem and are thus not capable of the kind of autonomy needed to achieve artificial general intelligence. Current models succeed at their tasks because people solve most of the problems to which these models are directed, leaving only simple computations for the model to perform, such as gradient descent. Another barrier is the need to recognize that there are multiple kinds of problems, some of which cannot be solved by available computational methods (for example, "insight problems"). Current methods for evaluating models (benchmarks and tests) are not adequate to identify the generality of the solutions, because it is impossible to infer the means by which a problem was solved from the fact of its solution. A test could be passed, for example, by a test-specific or a test-general method. It is a logical fallacy (affirming the consequent) to infer a method of solution from the observation of success. 

**Abstract (ZH)**: 当前和可预见的生成式AI模型无法实现人工通用智能，因为它们背负着人类债务。它们高度依赖人类输入来提供结构化的問題、架构和训练数据。它们将每个问题视为语言模式学习问题，因此无法具备实现人工通用智能所需的自主性。当前模型能够成功完成任务是因为人们基本上已经解决了这些模型所针对的问题，留给模型的只是简单的计算任务，例如梯度下降。另一个障碍是认识到存在多种类型的問題，其中一些问题无法通过现有的计算方法解决（例如，“洞察型问题”）。当前用于评估模型的方法（基准测试和测试）不足以识别解决方案的普遍性，因为无法从问题被解决的事实中推断出解决问题的方法。例如，一个测试可以通过特定于该测试的方法或者一般性的方法来通过。仅从成功的观察中推断解决方案的方法是一种逻辑谬误（肯定后件）。 

---
# Pre-Trained Video Generative Models as World Simulators 

**Title (ZH)**: 预训练视频生成模型作为世界模拟器 

**Authors**: Haoran He, Yang Zhang, Liang Lin, Zhongwen Xu, Ling Pan  

**Link**: [PDF](https://arxiv.org/pdf/2502.07825)  

**Abstract**: Video generative models pre-trained on large-scale internet datasets have achieved remarkable success, excelling at producing realistic synthetic videos. However, they often generate clips based on static prompts (e.g., text or images), limiting their ability to model interactive and dynamic scenarios. In this paper, we propose Dynamic World Simulation (DWS), a novel approach to transform pre-trained video generative models into controllable world simulators capable of executing specified action trajectories. To achieve precise alignment between conditioned actions and generated visual changes, we introduce a lightweight, universal action-conditioned module that seamlessly integrates into any existing model. Instead of focusing on complex visual details, we demonstrate that consistent dynamic transition modeling is the key to building powerful world simulators. Building upon this insight, we further introduce a motion-reinforced loss that enhances action controllability by compelling the model to capture dynamic changes more effectively. Experiments demonstrate that DWS can be versatilely applied to both diffusion and autoregressive transformer models, achieving significant improvements in generating action-controllable, dynamically consistent videos across games and robotics domains. Moreover, to facilitate the applications of the learned world simulator in downstream tasks such as model-based reinforcement learning, we propose prioritized imagination to improve sample efficiency, demonstrating competitive performance compared with state-of-the-art methods. 

**Abstract (ZH)**: 基于大型互联网数据集预训练的视频生成模型已取得显著成功，擅长生成逼真的合成视频。然而，它们通常根据静态提示（如文本或图像）生成片段，限制了其处理交互和动态场景的能力。本文提出了一种新颖的方法——动态世界模拟（DWS），旨在将预训练的视频生成模型转化为可控的世界模拟器，能够执行指定的动作轨迹。为实现条件动作与生成视觉变化的精确对齐，我们引入了一个轻量级、通用的动作条件模块，可以无缝集成到任何现有模型中。我们表明，一致的动态过渡建模是构建强大世界模拟器的关键。基于此见解，我们进一步引入了一种动力增强损失，通过促使模型更有效地捕捉动态变化来增强动作可控性。实验表明，DWS可以灵活应用于扩散和自回归变压器模型，实现显著改善的游戏和机器人领域中动作可控且动态一致的视频生成。此外，为促进所学世界模拟器在下游任务如模型基础强化学习中的应用，我们提出了优先想象以提高样本效率，并展示了与最新方法相当的性能。 

---
