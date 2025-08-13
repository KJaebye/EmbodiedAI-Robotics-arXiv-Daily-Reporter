# GeoVLA: Empowering 3D Representations in Vision-Language-Action Models 

**Title (ZH)**: GeoVLA：赋能视觉-语言-行动模型中的3D表示 

**Authors**: Lin Sun, Bin Xie, Yingfei Liu, Hao Shi, Tiancai Wang, Jiale Cao  

**Link**: [PDF](https://arxiv.org/pdf/2508.09071)  

**Abstract**: Vision-Language-Action (VLA) models have emerged as a promising approach for enabling robots to follow language instructions and predict corresponding this http URL, current VLA models mainly rely on 2D visual inputs, neglecting the rich geometric information in the 3D physical world, which limits their spatial awareness and adaptability. In this paper, we present GeoVLA, a novel VLA framework that effectively integrates 3D information to advance robotic manipulation. It uses a vision-language model (VLM) to process images and language instructions,extracting fused vision-language embeddings. In parallel, it converts depth maps into point clouds and employs a customized point encoder, called Point Embedding Network, to generate 3D geometric embeddings independently. These produced embeddings are then concatenated and processed by our proposed spatial-aware action expert, called 3D-enhanced Action Expert, which combines information from different sensor modalities to produce precise action sequences. Through extensive experiments in both simulation and real-world environments, GeoVLA demonstrates superior performance and robustness. It achieves state-of-the-art results in the LIBERO and ManiSkill2 simulation benchmarks and shows remarkable robustness in real-world tasks requiring height adaptability, scale awareness and viewpoint invariance. 

**Abstract (ZH)**: 基于3D信息的视觉-语言-动作（GeoVLA）框架：增强机器人操作的几何意识 

---
# Large Scale Robotic Material Handling: Learning, Planning, and Control 

**Title (ZH)**: 大规模机器人物料处理：学习、计划与控制 

**Authors**: Filippo A. Spinelli, Yifan Zhai, Fang Nan, Pascal Egli, Julian Nubert, Thilo Bleumer, Lukas Miller, Ferdinand Hofmann, Marco Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2508.09003)  

**Abstract**: Bulk material handling involves the efficient and precise moving of large quantities of materials, a core operation in many industries, including cargo ship unloading, waste sorting, construction, and demolition. These repetitive, labor-intensive, and safety-critical operations are typically performed using large hydraulic material handlers equipped with underactuated grippers. In this work, we present a comprehensive framework for the autonomous execution of large-scale material handling tasks. The system integrates specialized modules for environment perception, pile attack point selection, path planning, and motion control. The main contributions of this work are two reinforcement learning-based modules: an attack point planner that selects optimal grasping locations on the material pile to maximize removal efficiency and minimize the number of scoops, and a robust trajectory following controller that addresses the precision and safety challenges associated with underactuated grippers in movement, while utilizing their free-swinging nature to release material through dynamic throwing. We validate our framework through real-world experiments on a 40 t material handler in a representative worksite, focusing on two key tasks: high-throughput bulk pile management and high-precision truck loading. Comparative evaluations against human operators demonstrate the system's effectiveness in terms of precision, repeatability, and operational safety. To the best of our knowledge, this is the first complete automation of material handling tasks on a full scale. 

**Abstract (ZH)**: 大规模物料处理涉及大量材料的高效精准搬运，是许多行业，包括货物卸载、废物分类、建筑和拆除等核心操作。这些重复性强、劳动密集且安全性要求高的操作通常使用大型液压物料处理设备和欠驱动夹爪进行。本文提出了一种全面框架，用于自主执行大规模物料处理任务。该系统整合了专门用于环境感知、堆料攻击点选择、路径规划和运动控制的模块。本文的主要贡献是两个基于强化学习的模块：攻击点规划器，用于选择最大化移除效率并最小化铲斗次数的最佳抓取位置；以及鲁棒轨迹跟随控制器，该控制器解决了欠驱动夹爪在移动中面临的精确度和安全性挑战，并利用其自由摆动的特性通过动态投掷来释放物料。我们通过在典型工地上的40吨物料处理设备进行的实际实验验证了该框架，重点是两个关键任务：高通量堆料管理和高精度卡车装货。与人工操作者的对比评估表明，该系统在精确度、重复性和操作安全性方面具有显著效果。据我们所知，这是首次实现大规模物料处理任务的完全自动化。 

---
# Generation of Real-time Robotic Emotional Expressions Learning from Human Demonstration in Mixed Reality 

**Title (ZH)**: 实时机器人情感表达学习在混合现实中的人类示范教学 

**Authors**: Chao Wang, Michael Gienger, Fan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.08999)  

**Abstract**: Expressive behaviors in robots are critical for effectively conveying their emotional states during interactions with humans. In this work, we present a framework that autonomously generates realistic and diverse robotic emotional expressions based on expert human demonstrations captured in Mixed Reality (MR). Our system enables experts to teleoperate a virtual robot from a first-person perspective, capturing their facial expressions, head movements, and upper-body gestures, and mapping these behaviors onto corresponding robotic components including eyes, ears, neck, and arms. Leveraging a flow-matching-based generative process, our model learns to produce coherent and varied behaviors in real-time in response to moving objects, conditioned explicitly on given emotional states. A preliminary test validated the effectiveness of our approach for generating autonomous expressions. 

**Abstract (ZH)**: 机器人在与人类互动过程中有效传达情绪状态的表达行为至关重要。在本工作中，我们提出了一种基于混合现实（MR）中捕获的专家人类示范的框架，以自主生成真实且多样的机器人情感表达。该系统允许专家从第一人称视角远程操作虚拟机器人，捕获其面部表情、头部运动和上半身手势，并将这些行为映射到相应的机器人组件，包括眼睛、耳朵、颈部和手臂。利用基于流匹配的生成过程，我们的模型能够在实时中学习根据给定的情感状态生成一致且多样的行为。初步测试验证了该方法生成自主表达的有效性。 

---
# Rational Inverse Reasoning 

**Title (ZH)**: 理性逆向推理 

**Authors**: Ben Zandonati, Tomás Lozano-Pérez, Leslie Pack Kaelbling  

**Link**: [PDF](https://arxiv.org/pdf/2508.08983)  

**Abstract**: Humans can observe a single, imperfect demonstration and immediately generalize to very different problem settings. Robots, in contrast, often require hundreds of examples and still struggle to generalize beyond the training conditions. We argue that this limitation arises from the inability to recover the latent explanations that underpin intelligent behavior, and that these explanations can take the form of structured programs consisting of high-level goals, sub-task decomposition, and execution constraints. In this work, we introduce Rational Inverse Reasoning (RIR), a framework for inferring these latent programs through a hierarchical generative model of behavior. RIR frames few-shot imitation as Bayesian program induction: a vision-language model iteratively proposes structured symbolic task hypotheses, while a planner-in-the-loop inference scheme scores each by the likelihood of the observed demonstration under that hypothesis. This loop yields a posterior over concise, executable programs. We evaluate RIR on a suite of continuous manipulation tasks designed to test one-shot and few-shot generalization across variations in object pose, count, geometry, and layout. With as little as one demonstration, RIR infers the intended task structure and generalizes to novel settings, outperforming state-of-the-art vision-language model baselines. 

**Abstract (ZH)**: 人类可以从单一的不完美示范中立即泛化到非常不同的问题环境中，而机器人通常需要几百个示例，仍然难以在训练条件之外泛化。我们argue这一局限性源于无法恢复支撑智能行为的潜在解释，而这些解释可以表现为包含高层目标、子任务分解和执行约束的结构化程序。在本文中，我们引入了理性逆向推理（RIR）框架，通过行为的分层生成模型推断这些潜在程序。RIR将少量示范的模仿问题框架为贝叶斯程序归纳：一个视觉-语言模型迭代提出结构化符号任务假设，同时一个环路推理方案根据假设下观察到的示范发生的概率对每个假设进行评分。这个循环过程产生了一系列简洁且可执行的程序。我们在一系列旨在测试对象姿态、数量、几何形状和布局变化中的一次性和少量示范泛化能力的连续操作任务上评估了RIR。即使只有一个示范，RIR也能推断出预期的任务结构并泛化到新的环境中，优于最先进的视觉-语言模型基线。 

---
# Unsupervised Skill Discovery as Exploration for Learning Agile Locomotion 

**Title (ZH)**: 无监督技能发现作为探索学习敏捷移动的方法 

**Authors**: Seungeun Rho, Kartik Garg, Morgan Byrd, Sehoon Ha  

**Link**: [PDF](https://arxiv.org/pdf/2508.08982)  

**Abstract**: Exploration is crucial for enabling legged robots to learn agile locomotion behaviors that can overcome diverse obstacles. However, such exploration is inherently challenging, and we often rely on extensive reward engineering, expert demonstrations, or curriculum learning - all of which limit generalizability. In this work, we propose Skill Discovery as Exploration (SDAX), a novel learning framework that significantly reduces human engineering effort. SDAX leverages unsupervised skill discovery to autonomously acquire a diverse repertoire of skills for overcoming obstacles. To dynamically regulate the level of exploration during training, SDAX employs a bi-level optimization process that autonomously adjusts the degree of exploration. We demonstrate that SDAX enables quadrupedal robots to acquire highly agile behaviors including crawling, climbing, leaping, and executing complex maneuvers such as jumping off vertical walls. Finally, we deploy the learned policy on real hardware, validating its successful transfer to the real world. 

**Abstract (ZH)**: 探索对于使腿式机器人学会跨越各种障碍的敏捷运动行为至关重要。然而，这种探索本身具有挑战性，我们通常依赖于广泛的奖励工程、专家演示或 Curriculum 学习——所有这些都限制了泛化能力。在本文中，我们提出了一种称为技能发现作为探索（SDAX）的新学习框架，显著减少了人工工程努力。SDAX 利用无监督技能发现自主获取克服障碍所需的多样化技能库。为了在训练过程中动态调节探索程度，SDAX 采用了一种双层优化过程，能够自主调整探索程度。我们证明，SDAX 使四足机器人能够获得高度敏捷的行为，包括爬行、攀爬、跳跃以及执行复杂的操练，比如从垂直墙面上跳下。最后，我们在实际硬件上部署所学到的策略，验证了其成功地转移到现实世界的能力。 

---
# Towards Affordance-Aware Robotic Dexterous Grasping with Human-like Priors 

**Title (ZH)**: 面向 affordance 意识的类人先验驱动的灵巧抓取方法研究 

**Authors**: Haoyu Zhao, Linghao Zhuang, Xingyue Zhao, Cheng Zeng, Haoran Xu, Yuming Jiang, Jun Cen, Kexiang Wang, Jiayan Guo, Siteng Huang, Xin Li, Deli Zhao, Hua Zou  

**Link**: [PDF](https://arxiv.org/pdf/2508.08896)  

**Abstract**: A dexterous hand capable of generalizable grasping objects is fundamental for the development of general-purpose embodied AI. However, previous methods focus narrowly on low-level grasp stability metrics, neglecting affordance-aware positioning and human-like poses which are crucial for downstream manipulation. To address these limitations, we propose AffordDex, a novel framework with two-stage training that learns a universal grasping policy with an inherent understanding of both motion priors and object affordances. In the first stage, a trajectory imitator is pre-trained on a large corpus of human hand motions to instill a strong prior for natural movement. In the second stage, a residual module is trained to adapt these general human-like motions to specific object instances. This refinement is critically guided by two components: our Negative Affordance-aware Segmentation (NAA) module, which identifies functionally inappropriate contact regions, and a privileged teacher-student distillation process that ensures the final vision-based policy is highly successful. Extensive experiments demonstrate that AffordDex not only achieves universal dexterous grasping but also remains remarkably human-like in posture and functionally appropriate in contact location. As a result, AffordDex significantly outperforms state-of-the-art baselines across seen objects, unseen instances, and even entirely novel categories. 

**Abstract (ZH)**: 一种具备普适抓取能力的灵巧夹持器对于通用AGI的发展是基本的。之前的方法专注于低层次的抓取稳定性指标，忽略了知觉类适应性性和类人的姿态，为解决这些问题D我们我们提出了一种AffordD-DD框架D具备两阶段训练D能够学习到具有内在运动先先先和物体知知觉能力的抓取策略D第一阶段D轨迹模仿器预训练于大量基于人类手部运动的数据集以建立先先自然运动先D前D第二阶段D残差差生成训练DD适应于普D类类类人的运动到D特定DAD设定D这里的精细化由由步D由决定性的是两个模块D负D感知DD适应性分割D模块D识别功能不合适的区域DDD特权教师师生知识蒸馏过程D确保最终最终最终政策D高度成功D广泛的实验表明DDD不仅实现了普DD灵D的的抓取DDD姿态仍非常DD类人的D功能合适D因此DDAffordD-DD显著优于现有DD先进基准基线线素D对于未知D类类以前未见过类D和类全D类D分类 

---
# Robot can reduce superior's dominance in group discussions with human social hierarchy 

**Title (ZH)**: 机器人可以在包含人类社会等级结构的讨论中减少权威者的主导地位 yal。 

**Authors**: Kazuki Komura, Kumi Ozaki, Seiji Yamada  

**Link**: [PDF](https://arxiv.org/pdf/2508.08767)  

**Abstract**: This study investigated whether robotic agents that deal with social hierarchical relationships can reduce the dominance of superiors and equalize participation among participants in discussions with hierarchical structures. Thirty doctors and students having hierarchical relationship were gathered as participants, and an intervention experiment was conducted using a robot that can encourage participants to speak depending on social hierarchy. These were compared with strategies that intervened equally for all participants without considering hierarchy and with a no-action. The robots performed follow actions, showing backchanneling to speech, and encourage actions, prompting speech from members with less speaking time, on the basis of the hierarchical relationships among group members to equalize participation. The experimental results revealed that the robot's actions could potentially influence the speaking time among members, but it could not be conclusively stated that there were significant differences between the robot's action conditions. However, the results suggested that it might be possible to influence speaking time without decreasing the satisfaction of superiors. This indicates that in discussion scenarios where experienced superiors are likely to dominate, controlling the robot's backchanneling behavior could potentially suppress dominance and equalize participation among group members. 

**Abstract (ZH)**: 本研究探讨了处理社会层级关系的机器人代理是否能减少上级的支配地位并在具有层级结构的讨论中平等化参与者发言权。三十名具有层级关系的医生和学生作为参与者，进行了一项干预实验，使用能够根据社会层级鼓励参与者发言的机器人。这些实验结果被与均等地干预所有参与者、不考虑层级结构的无干预策略进行了对比。机器人根据组内成员之间的层级关系执行跟随行为（如回应发言）和促发表言行为，旨在平等化参与。实验结果表明，机器人的行为可能会影响成员的发言时间，但不能明确断定机器人干预条件之间存在显著差异。然而，结果提示在经验上级可能主导讨论的情景中，控制机器人的回应对话行为可能有助于抑制支配行为并平等化参与。 

---
# Visual Prompting for Robotic Manipulation with Annotation-Guided Pick-and-Place Using ACT 

**Title (ZH)**: 基于注释引导抓放操作的机器人操作视觉提示方法：ACT辅助 

**Authors**: Muhammad A. Muttaqien, Tomohiro Motoda, Ryo Hanai, Yukiyasu Domae  

**Link**: [PDF](https://arxiv.org/pdf/2508.08748)  

**Abstract**: Robotic pick-and-place tasks in convenience stores pose challenges due to dense object arrangements, occlusions, and variations in object properties such as color, shape, size, and texture. These factors complicate trajectory planning and grasping. This paper introduces a perception-action pipeline leveraging annotation-guided visual prompting, where bounding box annotations identify both pickable objects and placement locations, providing structured spatial guidance. Instead of traditional step-by-step planning, we employ Action Chunking with Transformers (ACT) as an imitation learning algorithm, enabling the robotic arm to predict chunked action sequences from human demonstrations. This facilitates smooth, adaptive, and data-driven pick-and-place operations. We evaluate our system based on success rate and visual analysis of grasping behavior, demonstrating improved grasp accuracy and adaptability in retail environments. 

**Abstract (ZH)**: 便利商店中机器人取放任务由于密集的物体排列、遮挡以及颜色、形状、大小和纹理等物体属性的差异而面临挑战。这些因素使得轨迹规划和抓取操作复杂化。本文提出一种基于注释引导的视觉提示感知-行动管道，通过边界框注解标识可拾取物体和放置位置，提供结构化的空间指导。不同于传统的分步规划，我们采用基于Transformer的动作片段化模仿学习算法（ACT），使机器人臂能够从人类示范中预测片断化动作序列。这促进了流畅、适应性和数据驱动的取放操作。我们根据抓取成功率和抓取行为的视觉分析评估系统，展示了在零售环境中的抓取准确性和适应性的提高。 

---
# Boosting Action-Information via a Variational Bottleneck on Unlabelled Robot Videos 

**Title (ZH)**: 通过变分瓶颈在未标记机器人视频上增强动作-信息传输 

**Authors**: Haoyu Zhang, Long Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.08743)  

**Abstract**: Learning from demonstrations (LfD) typically relies on large amounts of action-labeled expert trajectories, which fundamentally constrains the scale of available training data. A promising alternative is to learn directly from unlabeled video demonstrations. However, we find that existing methods tend to encode latent actions that share little mutual information with the true robot actions, leading to suboptimal control performance. To address this limitation, we introduce a novel framework that explicitly maximizes the mutual information between latent actions and true actions, even in the absence of action labels. Our method leverage the variational information-bottleneck to extract action-relevant representations while discarding task-irrelevant information. We provide a theoretical analysis showing that our objective indeed maximizes the mutual information between latent and true actions. Finally, we validate our approach through extensive experiments: first in simulated robotic environments and then on real-world robotic platforms, the experimental results demonstrate that our method significantly enhances mutual information and consistently improves policy performance. 

**Abstract (ZH)**: 从演示学习（LfD）通常依赖大量带动作标签的专家轨迹，这从根本上限制了可用训练数据的规模。一个有前景的替代方案是从未标记的视频演示中直接学习。然而，我们发现现有方法往往会编码与真实机器人动作共享少量互信息的潜在动作，导致次优控制性能。为解决这一局限，我们提出了一种新的框架，该框架显式地最大化潜在动作与真实动作之间的互信息，即使在缺乏动作标签的情况下也是如此。我们的方法利用变分信息瓶颈来提取与动作相关的信息同时摒弃与任务无关的信息。我们提供理论分析表明，我们的目标确实最大化了潜在动作与真实动作之间的互信息。最后，通过广泛的实验验证了我们的方法：首先在模拟机器人环境中，然后在真实世界机器人平台上，实验结果表明我们的方法显著提升了互信息并一致地提高了策略性能。 

---
# CRADLE: Conversational RTL Design Space Exploration with LLM-based Multi-Agent Systems 

**Title (ZH)**: CRriadge: 基于LLM的多智能体系统在对话式RTL设计空间探索中的应用 kuk 

**Authors**: Lukas Krupp, Maximilian Schöffel, Elias Biehl, Norbert Wehn  

**Link**: [PDF](https://arxiv.org/pdf/2508.08709)  

**Abstract**: This paper presents CRADLE, a conversational framework for design space exploration of RTL designs using LLM-based multi-agent systems. Unlike existing rigid approaches, CRADLE enables user-guided flows with internal self-verification, correction, and optimization. We demonstrate the framework with a generator-critic agent system targeting FPGA resource minimization using state-of-the-art LLMs. Experimental results on the RTLLM benchmark show that CRADLE achieves significant reductions in resource usage with averages of 48% and 40% in LUTs and FFs across all benchmark designs. 

**Abstract (ZH)**: CRADLE：基于LLM的多Agent系统在RTL设计空间探索中的对话框架 

---
# Towards Safe Imitation Learning via Potential Field-Guided Flow Matching 

**Title (ZH)**: 通过潜能场导向的流匹配实现安全的imitation学习 

**Authors**: Haoran Ding, Anqing Duan, Zezhou Sun, Leonel Rozo, Noémie Jaquier, Dezhen Song, Yoshihiko Nakamura  

**Link**: [PDF](https://arxiv.org/pdf/2508.08707)  

**Abstract**: Deep generative models, particularly diffusion and flow matching models, have recently shown remarkable potential in learning complex policies through imitation learning. However, the safety of generated motions remains overlooked, particularly in complex environments with inherent obstacles. In this work, we address this critical gap by proposing Potential Field-Guided Flow Matching Policy (PF2MP), a novel approach that simultaneously learns task policies and extracts obstacle-related information, represented as a potential field, from the same set of successful demonstrations. During inference, PF2MP modulates the flow matching vector field via the learned potential field, enabling safe motion generation. By leveraging these complementary fields, our approach achieves improved safety without compromising task success across diverse environments, such as navigation tasks and robotic manipulation scenarios. We evaluate PF2MP in both simulation and real-world settings, demonstrating its effectiveness in task space and joint space control. Experimental results demonstrate that PF2MP enhances safety, achieving a significant reduction of collisions compared to baseline policies. This work paves the way for safer motion generation in unstructured and obstaclerich environments. 

**Abstract (ZH)**: 潜在场引导的流匹配策略（PF2MP）：同时学习任务策略并提取障碍相关信息以实现安全运动生成 

---
# OmniVTLA: Vision-Tactile-Language-Action Model with Semantic-Aligned Tactile Sensing 

**Title (ZH)**: 全方位感官模型：与语义对齐的视觉-触觉-语言-行动模型 

**Authors**: Zhengxue Cheng, Yiqian Zhang, Wenkang Zhang, Haoyu Li, Keyu Wang, Li Song, Hengdi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.08706)  

**Abstract**: Recent vision-language-action (VLA) models build upon vision-language foundations, and have achieved promising results and exhibit the possibility of task generalization in robot manipulation. However, due to the heterogeneity of tactile sensors and the difficulty of acquiring tactile data, current VLA models significantly overlook the importance of tactile perception and fail in contact-rich tasks. To address this issue, this paper proposes OmniVTLA, a novel architecture involving tactile sensing. Specifically, our contributions are threefold. First, our OmniVTLA features a dual-path tactile encoder framework. This framework enhances tactile perception across diverse vision-based and force-based tactile sensors by using a pretrained vision transformer (ViT) and a semantically-aligned tactile ViT (SA-ViT). Second, we introduce ObjTac, a comprehensive force-based tactile dataset capturing textual, visual, and tactile information for 56 objects across 10 categories. With 135K tri-modal samples, ObjTac supplements existing visuo-tactile datasets. Third, leveraging this dataset, we train a semantically-aligned tactile encoder to learn a unified tactile representation, serving as a better initialization for OmniVTLA. Real-world experiments demonstrate substantial improvements over state-of-the-art VLA baselines, achieving 96.9% success rates with grippers, (21.9% higher over baseline) and 100% success rates with dexterous hands (6.2% higher over baseline) in pick-and-place tasks. Besides, OmniVTLA significantly reduces task completion time and generates smoother trajectories through tactile sensing compared to existing VLA. 

**Abstract (ZH)**: Recent Vision-LANGUAGE-Action (VLA) Models Incorporate Tactile Sensing for Task Generalization in Robot Manipulation: A Novel Architecture Named OmniVTLA 

---
# ZS-Puffin: Design, Modeling and Implementation of an Unmanned Aerial-Aquatic Vehicle with Amphibious Wings 

**Title (ZH)**: ZS-Puffin：兼具两栖翼的无人驾驶空水_vehicle的设计、建模与实现 

**Authors**: Zhenjiang Wang, Yunhua Jiang, Zikun Zhen, Yifan Jiang, Yubin Tan, Wubin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.08690)  

**Abstract**: Unmanned aerial-aquatic vehicles (UAAVs) can operate both in the air and underwater, giving them broad application prospects. Inspired by the dual-function wings of puffins, we propose a UAAV with amphibious wings to address the challenge posed by medium differences on the vehicle's propulsion system. The amphibious wing, redesigned based on a fixed-wing structure, features a single degree of freedom in pitch and requires no additional components. It can generate lift in the air and function as a flapping wing for propulsion underwater, reducing disturbance to marine life and making it environmentally friendly. Additionally, an artificial central pattern generator (CPG) is introduced to enhance the smoothness of the flapping motion. This paper presents the prototype, design details, and practical implementation of this concept. 

**Abstract (ZH)**: 无人驾驶水空两栖飞行器（UAAVs）可以在空中和水下操作，具有广泛的应用前景。受帝企鹅双功能翅膀的启发，我们提出了一种具有水空两栖翅膀的UAAV，以解决车辆推进系统在介质差异面前所面临的问题。基于固定翼结构重新设计的两栖翅膀仅具备俯仰单自由度，不需要额外组件，能够在空中产生升力，并在水下充当拍打翼进行推进，减少对海洋生命的干扰，使其更加环保。此外，引入人工中心模式生成器（CPG）以提高拍打运动的平滑度。本文介绍了该概念的原型、设计细节及其实际实施。 

---
# Communication Efficient Robotic Mixed Reality with Gaussian Splatting Cross-Layer Optimization 

**Title (ZH)**: 通信高效的机器人混合现实：高斯点绘制跨层优化 

**Authors**: Chenxuan Liu, He Li, Zongze Li, Shuai Wang, Wei Xu, Kejiang Ye, Derrick Wing Kwan Ng, Chengzhong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.08624)  

**Abstract**: Realizing low-cost communication in robotic mixed reality (RoboMR) systems presents a challenge, due to the necessity of uploading high-resolution images through wireless channels. This paper proposes Gaussian splatting (GS) RoboMR (GSMR), which enables the simulator to opportunistically render a photo-realistic view from the robot's pose by calling ``memory'' from a GS model, thus reducing the need for excessive image uploads. However, the GS model may involve discrepancies compared to the actual environments. To this end, a GS cross-layer optimization (GSCLO) framework is further proposed, which jointly optimizes content switching (i.e., deciding whether to upload image or not) and power allocation (i.e., adjusting to content profiles) across different frames by minimizing a newly derived GSMR loss function. The GSCLO problem is addressed by an accelerated penalty optimization (APO) algorithm that reduces computational complexity by over $10$x compared to traditional branch-and-bound and search algorithms. Moreover, variants of GSCLO are presented to achieve robust, low-power, and multi-robot GSMR. Extensive experiments demonstrate that the proposed GSMR paradigm and GSCLO method achieve significant improvements over existing benchmarks on both wheeled and legged robots in terms of diverse metrics in various scenarios. For the first time, it is found that RoboMR can be achieved with ultra-low communication costs, and mixture of data is useful for enhancing GS performance in dynamic scenarios. 

**Abstract (ZH)**: 低成本实现机器人混合现实（RoboMR）系统中的通信挑战：基于高斯点绘（GS）的RoboMR（GSMR）及跨层优化框架（GSCLO） 

---
# Autonomous Mobile Plant Watering Robot : A Kinematic Approach 

**Title (ZH)**: 自主移动植物浇水机器人：一种运动学方法 

**Authors**: Justin London  

**Link**: [PDF](https://arxiv.org/pdf/2508.08607)  

**Abstract**: Plants need regular and the appropriate amount of watering to thrive and survive. While agricultural robots exist that can spray water on plants and crops such as the , they are expensive and have limited mobility and/or functionality. We introduce a novel autonomous mobile plant watering robot that uses a 6 degree of freedom (DOF) manipulator, connected to a 4 wheel drive alloy chassis, to be able to hold a garden hose, recognize and detect plants, and to water them with the appropriate amount of water by being able to insert a soil humidity/moisture sensor into the soil. The robot uses Jetson Nano and Arduino microcontroller and real sense camera to perform computer vision to detect plants using real-time YOLOv5 with the Pl@ntNet-300K dataset. The robot uses LIDAR for object and collision avoideance and does not need to move on a pre-defined path and can keep track of which plants it has watered. We provide the Denavit-Hartenberg (DH) Table, forward kinematics, differential driving kinematics, and inverse kinematics along with simulation and experiment results 

**Abstract (ZH)**: 一种新型自主移动植物浇水机器人：使用6自由度机械臂和四轮驱动合金底盘进行植物识别与精确浇水 

---
# Developing a Calibrated Physics-Based Digital Twin for Construction Vehicles 

**Title (ZH)**: 基于物理原理的校准数字孪生技术在 construction vehicles 中的应用开发 

**Authors**: Deniz Karanfil, Daniel Lindmark, Martin Servin, David Torick, Bahram Ravani  

**Link**: [PDF](https://arxiv.org/pdf/2508.08576)  

**Abstract**: This paper presents the development of a calibrated digital twin of a wheel loader. A calibrated digital twin integrates a construction vehicle with a high-fidelity digital model allowing for automated diagnostics and optimization of operations as well as pre-planning simulations enhancing automation capabilities. The high-fidelity digital model is a virtual twin of the physical wheel loader. It uses a physics-based multibody dynamic model of the wheel loader in the software AGX Dynamics. Interactions of the wheel loader's bucket while in use in construction can be simulated in the virtual model. Calibration makes this simulation of high-fidelity which can enhance realistic planning for automation of construction operations. In this work, a wheel loader was instrumented with several sensors used to calibrate the digital model. The calibrated digital twin was able to estimate the magnitude of the forces on the bucket base with high accuracy, providing a high-fidelity simulation. 

**Abstract (ZH)**: 本文介绍了轮式装载机校准数字孪晶的开发。校准数字孪晶将施工车辆与高保真数字模型相结合，允许进行自动诊断和操作优化及增强自动化能力的预规划模拟。高保真数字模型是物理轮式装载机的虚拟孪生体。该模型在AGX Dynamics软件中使用基于物理的多体动力学模型进行轮式装载机的仿真。在使用过程中，轮式装载机铲斗的相互作用可以在虚拟模型中进行仿真。校准使这种仿真具有高保真度，可以提高施工操作自动化的现实规划能力。在本研究中，对轮式装载机安装了多个传感器以校准数字模型。校准后的数字孪晶能够以高精度估算铲斗底部的力，提供高保真度的仿真。 

---
# DeepFleet: Multi-Agent Foundation Models for Mobile Robots 

**Title (ZH)**: DeepFleet：移动机器人多智能体基础模型 

**Authors**: Ameya Agaskar, Sriram Siva, William Pickering, Kyle O'Brien, Charles Kekeh, Ang Li, Brianna Gallo Sarker, Alicia Chua, Mayur Nemade, Charun Thattai, Jiaming Di, Isaac Iyengar, Ramya Dharoor, Dino Kirouani, Jimmy Erskine, Tamir Hegazy, Scott Niekum, Usman A. Khan, Federico Pecora, Joseph W. Durham  

**Link**: [PDF](https://arxiv.org/pdf/2508.08574)  

**Abstract**: We introduce DeepFleet, a suite of foundation models designed to support coordination and planning for large-scale mobile robot fleets. These models are trained on fleet movement data, including robot positions, goals, and interactions, from hundreds of thousands of robots in Amazon warehouses worldwide. DeepFleet consists of four architectures that each embody a distinct inductive bias and collectively explore key points in the design space for multi-agent foundation models: the robot-centric (RC) model is an autoregressive decision transformer operating on neighborhoods of individual robots; the robot-floor (RF) model uses a transformer with cross-attention between robots and the warehouse floor; the image-floor (IF) model applies convolutional encoding to a multi-channel image representation of the full fleet; and the graph-floor (GF) model combines temporal attention with graph neural networks for spatial relationships. In this paper, we describe these models and present our evaluation of the impact of these design choices on prediction task performance. We find that the robot-centric and graph-floor models, which both use asynchronous robot state updates and incorporate the localized structure of robot interactions, show the most promise. We also present experiments that show that these two models can make effective use of larger warehouses operation datasets as the models are scaled up. 

**Abstract (ZH)**: DeepFleet：用于大规模移动机器人车队协调与规划的一套基础模型 

---
# AZRA: Extending the Affective Capabilities of Zoomorphic Robots using Augmented Reality 

**Title (ZH)**: AZRA：利用增强现实扩展拟人机器人的情感能力 

**Authors**: Shaun Macdonald, Salma ElSayed, Mark McGill  

**Link**: [PDF](https://arxiv.org/pdf/2508.08507)  

**Abstract**: Zoomorphic robots could serve as accessible and practical alternatives for users unable or unwilling to keep pets. However, their affective interactions are often simplistic and short-lived, limiting their potential for domestic adoption. In order to facilitate more dynamic and nuanced affective interactions and relationships between users and zoomorphic robots we present AZRA, a novel augmented reality (AR) framework that extends the affective capabilities of these robots without physical modifications. To demonstrate AZRA, we augment a zoomorphic robot, Petit Qoobo, with novel emotional displays (face, light, sound, thought bubbles) and interaction modalities (voice, touch, proximity, gaze). Additionally, AZRA features a computational model of emotion to calculate the robot's emotional responses, daily moods, evolving personality and needs. We highlight how AZRA can be used for rapid participatory prototyping and enhancing existing robots, then discuss implications on future zoomorphic robot development. 

**Abstract (ZH)**: 拟人化机器人可以为无法或不愿意养宠物的用户提供易用且实用的替代方案。然而，它们的情感交互往往过于简单且短暂，限制了其在家庭环境中的潜在应用。为了促进用户与拟人化机器人之间更富有动态性和层次性的情感交互和关系，我们提出了一种名为AZRA的新型增强现实（AR）框架，该框架可以通过软件扩展机器人的情感能力，而无需进行物理改造。为了展示AZRA，我们对拟人化机器人Petit Qoobo进行了增强，增加了新型的情感表现（面部、光线、声音、思维泡）和交互方式（语音、触觉、接近度、注视）。此外，AZRA还包含一个情感计算模型，用于计算机器人的情感反应、日常情绪、个性演变以及需求。我们强调AZRA在快速参与式原型制作和增强现有机器人方面的应用，然后讨论其对未来拟人化机器人发展的潜在影响。 

---
# A Minimal Model for Emergent Collective Behaviors in Autonomous Robotic Multi-Agent Systems 

**Title (ZH)**: 一种自主机器人多agent系统中 Emergent Collective Behaviors 的最小模型 

**Authors**: Hossein B. Jond  

**Link**: [PDF](https://arxiv.org/pdf/2508.08473)  

**Abstract**: Collective behaviors such as swarming and flocking emerge from simple, decentralized interactions in biological systems. Existing models, such as Vicsek and Cucker-Smale, lack collision avoidance, whereas the Olfati-Saber model imposes rigid formations, limiting their applicability in swarm robotics. To address these limitations, this paper proposes a minimal yet expressive model that governs agent dynamics using relative positions, velocities, and local density, modulated by two tunable parameters: the spatial offset and kinetic offset. The model achieves spatially flexible, collision-free behaviors that reflect naturalistic group dynamics. Furthermore, we extend the framework to cognitive autonomous systems, enabling energy-aware phase transitions between swarming and flocking through adaptive control parameter tuning. This cognitively inspired approach offers a robust foundation for real-world applications in multi-robot systems, particularly autonomous aerial swarms. 

**Abstract (ZH)**: 集体行为如集群和鸟群飞行源自生物系统中简单的分散交互。现有的模型如Vicsek和Cucker-Smale缺乏碰撞避免功能，而Olfati-Saber模型则限制了形成了刚性队形，限制了其在 swarm 机器人中的应用。为解决这些局限性，本文提出一个简洁但表达能力较强的模型，该模型利用相对位置、速度和局部密度来调控代理动态，并由两个可调参数：空间偏移和动能偏移来调制。该模型实现了空间灵活且无碰撞的行为，反映了自然群体动力学特征。此外，我们将该框架扩展到认知自主系统中，通过自适应控制参数调制实现集群和鸟群飞行之间的能效感知相变。这种受认知启发的方法为多机器人系统中的实际应用，特别是自主空中集群，提供了坚实的理论基础。 

---
# Whole-Body Coordination for Dynamic Object Grasping with Legged Manipulators 

**Title (ZH)**: 基于腿式 manipulator 的动态物体抓取的全身协调控制 

**Authors**: Qiwei Liang, Boyang Cai, Rongyi He, Hui Li, Tao Teng, Haihan Duan, Changxin Huang, Runhao Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2508.08328)  

**Abstract**: Quadrupedal robots with manipulators offer strong mobility and adaptability for grasping in unstructured, dynamic environments through coordinated whole-body control. However, existing research has predominantly focused on static-object grasping, neglecting the challenges posed by dynamic targets and thus limiting applicability in dynamic scenarios such as logistics sorting and human-robot collaboration. To address this, we introduce DQ-Bench, a new benchmark that systematically evaluates dynamic grasping across varying object motions, velocities, heights, object types, and terrain complexities, along with comprehensive evaluation metrics. Building upon this benchmark, we propose DQ-Net, a compact teacher-student framework designed to infer grasp configurations from limited perceptual cues. During training, the teacher network leverages privileged information to holistically model both the static geometric properties and dynamic motion characteristics of the target, and integrates a grasp fusion module to deliver robust guidance for motion planning. Concurrently, we design a lightweight student network that performs dual-viewpoint temporal modeling using only the target mask, depth map, and proprioceptive state, enabling closed-loop action outputs without reliance on privileged data. Extensive experiments on DQ-Bench demonstrate that DQ-Net achieves robust dynamic objects grasping across multiple task settings, substantially outperforming baseline methods in both success rate and responsiveness. 

**Abstract (ZH)**: 四足 manipulator 机器人通过协调全身控制在未结构化和动态环境中提供了强大的移动性和适应性，以实现抓取。然而，现有研究主要集中在静态物体抓取上，忽视了动态目标所带来的挑战，从而限制了其在物流分拣和人机协作等动态场景中的应用。为解决这一问题，我们引入了 DQ-Bench，这是一个新的基准系统地评估不同物体运动、速度、高度、物体类型和地形复杂度下的动态抓取，并提供了全面的评估指标。基于这一基准，我们提出了一种紧凑的教师-学生框架 DQ-Net，用于从有限的感知线索中推断抓取配置。在训练过程中，教师网络利用专属信息整体建模目标的静态几何特性和动态运动特征，并集成了一个抓取融合模块，以提供稳健的运动规划指导。同时，我们设计了一个轻量级的学生网络，仅使用目标掩码、深度图和本体内省状态进行双视角时序建模，从而实现闭环动作输出，无需依赖专属数据。在 DQ-Bench 上进行的大量实验表明，DQ-Net 在多种任务设置下实现了稳健的动态物体抓取，其成功率和响应性显著优于Baseline方法。 

---
# Evaluation of an Autonomous Surface Robot Equipped with a Transformable Mobility Mechanism for Efficient Mobility Control 

**Title (ZH)**: 装有可变运动机制的自主水面机器人高效运动控制评估 

**Authors**: Yasuyuki Fujii, Dinh Tuan Tran, Joo-Ho Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.08303)  

**Abstract**: Efficient mobility and power consumption are critical for autonomous water surface robots in long-term water environmental monitoring. This study develops and evaluates a transformable mobility mechanism for a water surface robot with two control modes: station-keeping and traveling to improve energy efficiency and maneuverability. Field experiments show that, in a round-trip task between two points, the traveling mode reduces power consumption by 10\% and decreases the total time required for travel by 5\% compared to the station-keeping mode. These results confirm the effectiveness of the transformable mobility mechanism for enhancing operational efficiency in patrolling on water surface. 

**Abstract (ZH)**: 高效的移动能力和功率消耗对于自主水面机器人在长期水环境监测中的应用至关重要。本研究开发并评估了一种可变形移动机制，该机制适用于具有站保持和航行两种控制模式的水面机器人，以提高能源效率和机动性。实地实验表明，在两点之间的往返任务中，航行模式相较于站保持模式可减少功率消耗10%并降低总航行时间5%。这些结果证实了可变形移动机制在水面巡逻操作效率提升方面的有效性。 

---
# emg2tendon: From sEMG Signals to Tendon Control in Musculoskeletal Hands 

**Title (ZH)**: emg2tendon: 从表面肌电信号到肌腱控制在肌骨骼手中的应用 

**Authors**: Sagar Verma  

**Link**: [PDF](https://arxiv.org/pdf/2508.08269)  

**Abstract**: Tendon-driven robotic hands offer unparalleled dexterity for manipulation tasks, but learning control policies for such systems presents unique challenges. Unlike joint-actuated robotic hands, tendon-driven systems lack a direct one-to-one mapping between motion capture (mocap) data and tendon controls, making the learning process complex and expensive. Additionally, visual tracking methods for real-world applications are prone to occlusions and inaccuracies, further complicating joint tracking. Wrist-wearable surface electromyography (sEMG) sensors present an inexpensive, robust alternative to capture hand motion. However, mapping sEMG signals to tendon control remains a significant challenge despite the availability of EMG-to-pose data sets and regression-based models in the existing literature.
We introduce the first large-scale EMG-to-Tendon Control dataset for robotic hands, extending the emg2pose dataset, which includes recordings from 193 subjects, spanning 370 hours and 29 stages with diverse gestures. This dataset incorporates tendon control signals derived using the MyoSuite MyoHand model, addressing limitations such as invalid poses in prior methods. We provide three baseline regression models to demonstrate emg2tendon utility and propose a novel diffusion-based regression model for predicting tendon control from sEMG recordings. This dataset and modeling framework marks a significant step forward for tendon-driven dexterous robotic manipulation, laying the groundwork for scalable and accurate tendon control in robotic hands. this https URL 

**Abstract (ZH)**: 肌腱驱动的机器人手在操作任务中提供了无与伦比的灵活性，但学习such系统的控制策略带来了独特的挑战。与关节驱动的机器人手不同，肌腱驱动系统缺乏运动捕捉（mocap）数据与肌腱控制之间的直接一对一映射关系，使得学习过程复杂且昂贵。此外，现实世界应用中的视觉跟踪方法容易受到遮挡和不准确的影响，进一步增加了关节跟踪的复杂性。腕戴式表面肌电图（sEMG）传感器提供了一种低成本且稳健的替代方法来捕捉手部运动。尽管已有文献中存在肌电图（EMG）到姿势的数据集和基于回归的模型，将sEMG信号映射到肌腱控制仍然是一个重要挑战。 

---
# Forecast-Driven MPC for Decentralized Multi-Robot Collision Avoidance 

**Title (ZH)**: 基于预测的分散多机器人碰撞 avoidance 控制 

**Authors**: Hadush Hailu, Bruk Gebregziabher, Prudhvi Raj  

**Link**: [PDF](https://arxiv.org/pdf/2508.08264)  

**Abstract**: The Iterative Forecast Planner (IFP) is a geometric planning approach that offers lightweight computations, scal- able, and reactive solutions for multi-robot path planning in decentralized, communication-free settings. However, it struggles in symmetric configurations, where mirrored interactions often lead to collisions and deadlocks. We introduce eIFP-MPC, an optimized and extended version of IFP that improves robustness and path consistency in dense, dynamic environments. The method refines threat prioritization using a time-to-collision heuristic, stabilizes path generation through cost-based via- point selection, and ensures dynamic feasibility by incorporating model predictive control (MPC) into the planning process. These enhancements are tightly integrated into the IFP to preserve its efficiency while improving its adaptability and stability. Ex- tensive simulations across symmetric and high-density scenarios show that eIFP-MPC significantly reduces oscillations, ensures collision-free motion, and improves trajectory efficiency. The results demonstrate that geometric planners can be strengthened through optimization, enabling robust performance at scale in complex multi-agent environments. 

**Abstract (ZH)**: 基于迭代预测规划者的增强多机器人路径规划方法(eIFP-MPC) 

---
# Koopman Operator Based Linear Model Predictive Control for Quadruped Trotting 

**Title (ZH)**: 基于科氏算子的线性模型预测控制四足足动捷步 

**Authors**: Chun-Ming Yang, Pranav A. Bhounsule  

**Link**: [PDF](https://arxiv.org/pdf/2508.08259)  

**Abstract**: Online optimal control of quadruped robots would enable them to adapt to varying inputs and changing conditions in real time. A common way of achieving this is linear model predictive control (LMPC), where a quadratic programming (QP) problem is formulated over a finite horizon with a quadratic cost and linear constraints obtained by linearizing the equations of motion and solved on the fly. However, the model linearization may lead to model inaccuracies. In this paper, we use the Koopman operator to create a linear model of the quadrupedal system in high dimensional space which preserves the nonlinearity of the equations of motion. Then using LMPC, we demonstrate high fidelity tracking and disturbance rejection on a quadrupedal robot. This is the first work that uses the Koopman operator theory for LMPC of quadrupedal locomotion. 

**Abstract (ZH)**: 基于科恩曼算子的四足机器人在线最优控制：高保真跟踪与扰动 rejection 

---
# Humanoid Robot Acrobatics Utilizing Complete Articulated Rigid Body Dynamics 

**Title (ZH)**: 利用完整刚体 articulated 动力学的人形机器人杂技技巧 

**Authors**: Gerald Brantner  

**Link**: [PDF](https://arxiv.org/pdf/2508.08258)  

**Abstract**: Endowing humanoid robots with the ability to perform highly dynamic motions akin to human-level acrobatics has been a long-standing challenge. Successfully performing these maneuvers requires close consideration of the underlying physics in both trajectory optimization for planning and control during execution. This is particularly challenging due to humanoids' high degree-of-freedom count and associated exponentially scaling complexities, which makes planning on the explicit equations of motion intractable. Typical workarounds include linearization methods and model approximations. However, neither are sufficient because they produce degraded performance on the true robotic system. This paper presents a control architecture comprising trajectory optimization and whole-body control, intermediated by a matching model abstraction, that enables the execution of acrobatic maneuvers, including constraint and posture behaviors, conditioned on the unabbreviated equations of motion of the articulated rigid body model. A review of underlying modeling and control methods is given, followed by implementation details including model abstraction, trajectory optimization and whole-body controller. The system's effectiveness is analyzed in simulation. 

**Abstract (ZH)**: 赋予类人机器人执行类似人类杂技的高动态动作的能力是一个长期存在的挑战。成功执行这些动作需要在轨迹优化计划和执行控制中密切考虑其背后的物理学。由于类人机器人具有很高的自由度和相应的指数级复杂性，使运动规划适用于显式的运动方程无法实现。常见的解决方法包括线性化方法和模型近似。然而，这两种方法都不足以在真正的机器人系统中产生满意的性能。本文提出了一种控制架构，包括轨迹优化和全身控制，并通过匹配模型抽象中介，能够在 articulated 刚体模型完整运动方程约束下执行杂技动作，包括约束和姿态行为。文中提供了基本建模和控制方法的回顾，以及模型抽象、轨迹优化和全身控制器的实现细节。系统的效果在仿真中进行了分析。 

---
# Spatial Traces: Enhancing VLA Models with Spatial-Temporal Understanding 

**Title (ZH)**: 空间踪迹：增强VLA模型的空间-时间理解能力 

**Authors**: Maxim A. Patratskiy, Alexey K. Kovalev, Aleksandr I. Panov  

**Link**: [PDF](https://arxiv.org/pdf/2508.09032)  

**Abstract**: Vision-Language-Action models have demonstrated remarkable capabilities in predicting agent movements within virtual environments and real-world scenarios based on visual observations and textual instructions. Although recent research has focused on enhancing spatial and temporal understanding independently, this paper presents a novel approach that integrates both aspects through visual prompting. We introduce a method that projects visual traces of key points from observations onto depth maps, enabling models to capture both spatial and temporal information simultaneously. The experiments in SimplerEnv show that the mean number of tasks successfully solved increased for 4% compared to SpatialVLA and 19% compared to TraceVLA. Furthermore, we show that this enhancement can be achieved with minimal training data, making it particularly valuable for real-world applications where data collection is challenging. The project page is available at this https URL. 

**Abstract (ZH)**: 视觉-语言-行动模型在基于视觉观察和文本指令预测代理在虚拟环境和现实世界场景中动作方面展示了显著的能力。尽管近期研究专注于独立提升空间和时间理解，本文提出了一种通过视觉提示将两者结合起来的新方法。我们提出了一种将观察中关键点的视觉轨迹投影到深度图上的方法，从而使模型能够同时捕捉空间和时间信息。在SimplerEnv的实验中，与SpatialVLA相比，成功解决的任务平均数量增加了4%，与TraceVLA相比增加了19%。此外，我们展示了这一增强可以在最少的训练数据下实现，使其特别适合于数据收集具有挑战性的现实世界应用。更多信息请访问这个网址：https URL。 

---
# Shape Completion and Real-Time Visualization in Robotic Ultrasound Spine Acquisitions 

**Title (ZH)**: 机器人超声脊柱检查中的形状完成与实时可视化 

**Authors**: Miruna-Alexandra Gafencu, Reem Shaban, Yordanka Velikova, Mohammad Farid Azampour, Nassir Navab  

**Link**: [PDF](https://arxiv.org/pdf/2508.08923)  

**Abstract**: Ultrasound (US) imaging is increasingly used in spinal procedures due to its real-time, radiation-free capabilities; however, its effectiveness is hindered by shadowing artifacts that obscure deeper tissue structures. Traditional approaches, such as CT-to-US registration, incorporate anatomical information from preoperative CT scans to guide interventions, but they are limited by complex registration requirements, differences in spine curvature, and the need for recent CT imaging. Recent shape completion methods can offer an alternative by reconstructing spinal structures in US data, while being pretrained on large set of publicly available CT scans. However, these approaches are typically offline and have limited reproducibility. In this work, we introduce a novel integrated system that combines robotic ultrasound with real-time shape completion to enhance spinal visualization. Our robotic platform autonomously acquires US sweeps of the lumbar spine, extracts vertebral surfaces from ultrasound, and reconstructs the complete anatomy using a deep learning-based shape completion network. This framework provides interactive, real-time visualization with the capability to autonomously repeat scans and can enable navigation to target locations. This can contribute to better consistency, reproducibility, and understanding of the underlying anatomy. We validate our approach through quantitative experiments assessing shape completion accuracy and evaluations of multiple spine acquisition protocols on a phantom setup. Additionally, we present qualitative results of the visualization on a volunteer scan. 

**Abstract (ZH)**: 基于超声成像的脊柱手术成像系统：结合机器人超声与实时形状补全 

---
# DiffPhysCam: Differentiable Physics-Based Camera Simulation for Inverse Rendering and Embodied AI 

**Title (ZH)**: DiffPhysCam: 基于可微物理的相机仿真及其在逆向渲染和具身AI中的应用 

**Authors**: Bo-Hsun Chen, Nevindu M. Batagoda, Dan Negrut  

**Link**: [PDF](https://arxiv.org/pdf/2508.08831)  

**Abstract**: We introduce DiffPhysCam, a differentiable camera simulator designed to support robotics and embodied AI applications by enabling gradient-based optimization in visual perception pipelines. Generating synthetic images that closely mimic those from real cameras is essential for training visual models and enabling end-to-end visuomotor learning. Moreover, differentiable rendering allows inverse reconstruction of real-world scenes as digital twins, facilitating simulation-based robotics training. However, existing virtual cameras offer limited control over intrinsic settings, poorly capture optical artifacts, and lack tunable calibration parameters -- hindering sim-to-real transfer. DiffPhysCam addresses these limitations through a multi-stage pipeline that provides fine-grained control over camera settings, models key optical effects such as defocus blur, and supports calibration with real-world data. It enables both forward rendering for image synthesis and inverse rendering for 3D scene reconstruction, including mesh and material texture optimization. We show that DiffPhysCam enhances robotic perception performance in synthetic image tasks. As an illustrative example, we create a digital twin of a real-world scene using inverse rendering, simulate it in a multi-physics environment, and demonstrate navigation of an autonomous ground vehicle using images generated by DiffPhysCam. 

**Abstract (ZH)**: DiffPhysCam：一种用于支持机器人和体现AI应用的可微摄像头模拟器 

---
# Decoupling Geometry from Optimization in 2D Irregular Cutting and Packing Problems: an Open-Source Collision Detection Engine 

**Title (ZH)**: 在二维不规则切割与排版问题中解耦几何与优化：一个开源碰撞检测引擎 

**Authors**: Jeroen Gardeyn, Tony Wauters, Greet Vanden Berghe  

**Link**: [PDF](https://arxiv.org/pdf/2508.08341)  

**Abstract**: Addressing irregular cutting and packing (C&P) optimization problems poses two distinct challenges: the geometric challenge of determining whether or not an item can be placed feasibly at a certain position, and the optimization challenge of finding a good solution according to some objective function. Until now, those tackling such problems have had to address both challenges simultaneously, requiring two distinct sets of expertise and a lot of research & development effort. One way to lower this barrier is to decouple the two challenges. In this paper we introduce a powerful collision detection engine (CDE) for 2D irregular C&P problems which assumes full responsibility for the geometric challenge. The CDE (i) allows users to focus with full confidence on their optimization challenge by abstracting geometry away and (ii) enables independent advances to propagate to all optimization algorithms built atop it. We present a set of core principles and design philosophies to model a general and adaptable CDE focused on maximizing performance, accuracy and robustness. These principles are accompanied by a concrete open-source implementation called $\texttt{jagua-rs}$. This paper together with its implementation serves as a catalyst for future advances in irregular C&P problems by providing a solid foundation which can either be used as it currently exists or be further improved upon. 

**Abstract (ZH)**: 解决非规则切割与排样（C&P）优化问题面临着两个独特的挑战：几何挑战，即确定项目是否可以放置在某个位置，以及优化挑战，即根据某些目标函数找到一个好的解决方案。直到现在，解决这些问题的研究者不得不同时应对这两个挑战，这需要两种不同的专长和技术研发努力。一种降低这种障碍的方法是将这两个挑战分离。本文介绍了一种强大的二维非规则C&P问题碰撞检测引擎（CDE），该引擎完全负责解决几何挑战。CDE能够通过抽象掉几何问题，让用户全神贯注于优化挑战，同时允许独立的技术进步传递给所有基于它的优化算法。本文提出了一套核心原则和设计哲学，旨在构建一个通用且可适应的CDE，以最大化性能、准确性和稳健性。这些原则伴随着一个具体的开源实现，名为$\texttt{jagua-rs}$。本文与其实现共同作用，为未来非规则C&P问题的进步提供了一个坚实的基础，既可以现用也可进一步改进。 

---
# MinionsLLM: a Task-adaptive Framework For The Training and Control of Multi-Agent Systems Through Natural Language 

**Title (ZH)**: MinionsLLM：一种通过自然语言进行多agent系统训练和控制的任务自适应框架 

**Authors**: Andres Garcia Rincon, Eliseo Ferrante  

**Link**: [PDF](https://arxiv.org/pdf/2508.08283)  

**Abstract**: This paper presents MinionsLLM, a novel framework that integrates Large Language Models (LLMs) with Behavior Trees (BTs) and Formal Grammars to enable natural language control of multi-agent systems within arbitrary, user-defined environments. MinionsLLM provides standardized interfaces for defining environments, agents, and behavioral primitives, and introduces two synthetic dataset generation methods (Method A and Method B) to fine-tune LLMs for improved syntactic validity and semantic task relevance. We validate our approach using Google's Gemma 3 model family at three parameter scales (1B, 4B, and 12B) and demonstrate substantial gains: Method B increases syntactic validity to 92.6% and achieves a mean task performance improvement of 33% over baseline. Notably, our experiments show that smaller models benefit most from fine-tuning, suggesting promising directions for deploying compact, locally hosted LLMs in resource-constrained multi-agent control scenarios. The framework and all resources are released open-source to support reproducibility and future research. 

**Abstract (ZH)**: MinionsLLM：一种将大型语言模型与行为树和形式语法集成的新框架，以在任意用户定义环境中控制多Agent系统 

---
# Where is the Boundary: Multimodal Sensor Fusion Test Bench for Tissue Boundary Delineation 

**Title (ZH)**: 边界何在：多模态传感器融合测试平台用于组织边界勾画 

**Authors**: Zacharias Chen, Alexa Cristelle Cahilig, Sarah Dias, Prithu Kolar, Ravi Prakash, Patrick J. Codd  

**Link**: [PDF](https://arxiv.org/pdf/2508.08257)  

**Abstract**: Robot-assisted neurological surgery is receiving growing interest due to the improved dexterity, precision, and control of surgical tools, which results in better patient outcomes. However, such systems often limit surgeons' natural sensory feedback, which is crucial in identifying tissues -- particularly in oncological procedures where distinguishing between healthy and tumorous tissue is vital. While imaging and force sensing have addressed the lack of sensory feedback, limited research has explored multimodal sensing options for accurate tissue boundary delineation. We present a user-friendly, modular test bench designed to evaluate and integrate complementary multimodal sensors for tissue identification. Our proposed system first uses vision-based guidance to estimate boundary locations with visual cues, which are then refined using data acquired by contact microphones and a force sensor. Real-time data acquisition and visualization are supported via an interactive graphical interface. Experimental results demonstrate that multimodal fusion significantly improves material classification accuracy. The platform provides a scalable hardware-software solution for exploring sensor fusion in surgical applications and demonstrates the potential of multimodal approaches in real-time tissue boundary delineation. 

**Abstract (ZH)**: 辅助神经外科手术的机器人系统因其实现了更高的灵巧性、精确度和控制性，从而改善了手术效果，正逐渐受到关注。然而，这样的系统往往限制了外科医生的自然感官反馈，这对于识别组织至关重要，尤其是在区分健康组织与肿瘤组织的肿瘤外科手术中尤为重要。虽然成像和力感知已弥补了感官反馈的不足，但鲜有研究探讨多模态感知选项以实现准确的组织边界识别。我们提出了一种用户友好、模块化的测试平台，用于评估和集成互补的多模态传感器以识别组织。该系统首先使用基于视觉的引导来通过视觉提示估计边界位置，然后利用接触麦克风和力传感器获取的数据进行细化。通过交互式图形界面实现了实时数据采集与可视化。实验结果表明，多模态融合显著提高了材料分类准确性。该平台提供了可扩展的硬件-软件解决方案，用于在手术应用中探索传感器融合，并展示了实时组织边界分割中多模态方法的潜力。 

---
