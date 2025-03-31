# Robust Offline Imitation Learning Through State-level Trajectory Stitching 

**Title (ZH)**: 基于状态级轨迹拼接的鲁棒离线 imitation 学习 

**Authors**: Shuze Wang, Yunpeng Mei, Hongjie Cao, Yetian Yuan, Gang Wang, Jian Sun, Jie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.22524)  

**Abstract**: Imitation learning (IL) has proven effective for enabling robots to acquire visuomotor skills through expert demonstrations. However, traditional IL methods are limited by their reliance on high-quality, often scarce, expert data, and suffer from covariate shift. To address these challenges, recent advances in offline IL have incorporated suboptimal, unlabeled datasets into the training. In this paper, we propose a novel approach to enhance policy learning from mixed-quality offline datasets by leveraging task-relevant trajectory fragments and rich environmental dynamics. Specifically, we introduce a state-based search framework that stitches state-action pairs from imperfect demonstrations, generating more diverse and informative training trajectories. Experimental results on standard IL benchmarks and real-world robotic tasks showcase that our proposed method significantly improves both generalization and performance. 

**Abstract (ZH)**: 基于任务相关轨迹片段和丰富环境动力学的混合质量离线 imitation 学习方法 

---
# Scenario Dreamer: Vectorized Latent Diffusion for Generating Driving Simulation Environments 

**Title (ZH)**: 情景梦师：向量化的潜空间扩散模型用于生成驾驶模拟环境 

**Authors**: Luke Rowe, Roger Girgis, Anthony Gosselin, Liam Paull, Christopher Pal, Felix Heide  

**Link**: [PDF](https://arxiv.org/pdf/2503.22496)  

**Abstract**: We introduce Scenario Dreamer, a fully data-driven generative simulator for autonomous vehicle planning that generates both the initial traffic scene - comprising a lane graph and agent bounding boxes - and closed-loop agent behaviours. Existing methods for generating driving simulation environments encode the initial traffic scene as a rasterized image and, as such, require parameter-heavy networks that perform unnecessary computation due to many empty pixels in the rasterized scene. Moreover, we find that existing methods that employ rule-based agent behaviours lack diversity and realism. Scenario Dreamer instead employs a novel vectorized latent diffusion model for initial scene generation that directly operates on the vectorized scene elements and an autoregressive Transformer for data-driven agent behaviour simulation. Scenario Dreamer additionally supports scene extrapolation via diffusion inpainting, enabling the generation of unbounded simulation environments. Extensive experiments show that Scenario Dreamer outperforms existing generative simulators in realism and efficiency: the vectorized scene-generation base model achieves superior generation quality with around 2x fewer parameters, 6x lower generation latency, and 10x fewer GPU training hours compared to the strongest baseline. We confirm its practical utility by showing that reinforcement learning planning agents are more challenged in Scenario Dreamer environments than traditional non-generative simulation environments, especially on long and adversarial driving environments. 

**Abstract (ZH)**: 情景梦者：一种全数据驱动的自主车辆规划生成模拟器 

---
# Cooperative Hybrid Multi-Agent Pathfinding Based on Shared Exploration Maps 

**Title (ZH)**: 基于共享探索地图的合作混合多智能体路径规划 

**Authors**: Ning Liu, Sen Shen, Xiangrui Kong, Hongtao Zhang, Thomas Bräunl  

**Link**: [PDF](https://arxiv.org/pdf/2503.22162)  

**Abstract**: Multi-Agent Pathfinding is used in areas including multi-robot formations, warehouse logistics, and intelligent vehicles. However, many environments are incomplete or frequently change, making it difficult for standard centralized planning or pure reinforcement learning to maintain both global solution quality and local flexibility. This paper introduces a hybrid framework that integrates D* Lite global search with multi-agent reinforcement learning, using a switching mechanism and a freeze-prevention strategy to handle dynamic conditions and crowded settings. We evaluate the framework in the discrete POGEMA environment and compare it with baseline methods. Experimental outcomes indicate that the proposed framework substantially improves success rate, collision rate, and path efficiency. The model is further tested on the EyeSim platform, where it maintains feasible Pathfinding under frequent changes and large-scale robot deployments. 

**Abstract (ZH)**: 多智能体路径规划在多机器人编队、仓库物流和智能车辆等领域被广泛应用。然而，许多环境不完整或频繁变化，使得标准的集中规划或纯强化学习难以同时保持全局解的质量和局部灵活性。本文提出了一种将D* Lite全局搜索与多智能体强化学习相结合的混合框架，通过切换机制和防冻结策略处理动态条件和密集环境。我们在离散POGEMA环境中评估了该框架，并与基线方法进行比较。实验结果表明，所提出的框架显著提高了成功率、碰撞率和路径效率。该模型进一步在EyeSim平台上测试，能够在频繁变化和大规模机器人部署的环境中保持有效的路径规划。 

---
# Unicorn: Text-Only Data Synthesis for Vision Language Model Training 

**Title (ZH)**: 独角兽：仅文本数据合成用于视觉语言模型训练 

**Authors**: Xiaomin Yu, Pengxiang Ding, Wenjie Zhang, Siteng Huang, Songyang Gao, Chengwei Qin, Kejian Wu, Zhaoxin Fan, Ziyue Qiao, Donglin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.22655)  

**Abstract**: Training vision-language models (VLMs) typically requires large-scale, high-quality image-text pairs, but collecting or synthesizing such data is costly. In contrast, text data is abundant and inexpensive, prompting the question: can high-quality multimodal training data be synthesized purely from text? To tackle this, we propose a cross-integrated three-stage multimodal data synthesis framework, which generates two datasets: Unicorn-1.2M and Unicorn-471K-Instruction. In Stage 1: Diverse Caption Data Synthesis, we construct 1.2M semantically diverse high-quality captions by expanding sparse caption seeds using large language models (LLMs). In Stage 2: Instruction-Tuning Data Generation, we further process 471K captions into multi-turn instruction-tuning tasks to support complex reasoning. Finally, in Stage 3: Modality Representation Transfer, these textual captions representations are transformed into visual representations, resulting in diverse synthetic image representations. This three-stage process enables us to construct Unicorn-1.2M for pretraining and Unicorn-471K-Instruction for instruction-tuning, without relying on real images. By eliminating the dependency on real images while maintaining data quality and diversity, our framework offers a cost-effective and scalable solution for VLMs training. Code is available at this https URL. 

**Abstract (ZH)**: 训练视觉-语言模型（VLMs）通常需要大规模的高质量图像-文本配对数据，但收集或合成这样的数据成本高昂。相比之下，文本数据丰富且成本低廉，这促使我们问：是否可以纯粹从文本中合成高质量的多模态训练数据？为解决这一问题，我们提出了一种跨模态三阶段数据合成框架，生成两个数据集：Unicorn-1.2M和Unicorn-471K-Instruction。在第一阶段：多元化标题数据合成中，我们使用大型语言模型（LLMs）扩展稀疏的标题种子，构造出1.2M个语义多样的高质量标题。在第二阶段：指令调优数据生成中，我们将471K个标题进一步处理成多轮指令调优任务，以支持复杂的推理。最后，在第三阶段：模态表示迁移中，这些文本标题表示被转换为视觉表示，生成多样的合成图像表示。这一三阶段过程使我们能够在不依赖真实图像的情况下构建Unicorn-1.2M用于预训练和Unicorn-471K-Instruction用于指令调优。通过消除对真实图像的依赖性，同时保持数据质量和多样性，我们的框架为VLMs的训练提供了成本效益高且可扩展的解决方案。代码详见this https URL。 

---
# CPPO: Accelerating the Training of Group Relative Policy Optimization-Based Reasoning Models 

**Title (ZH)**: CPPO：加速基于群组相对策略优化的推理模型训练 

**Authors**: Zhihang Lin, Mingbao Lin, Yuan Xie, Rongrong Ji  

**Link**: [PDF](https://arxiv.org/pdf/2503.22342)  

**Abstract**: This paper introduces Completion Pruning Policy Optimization (CPPO) to accelerate the training of reasoning models based on Group Relative Policy Optimization (GRPO). GRPO, while effective, incurs high training costs due to the need for sampling multiple completions for each question. Our experiment and theoretical analysis reveals that the number of completions impacts model accuracy yet increases training time multiplicatively, and not all completions contribute equally to policy training -- their contribution depends on their relative advantage. To address these issues, we propose CPPO, which prunes completions with low absolute advantages, significantly reducing the number needed for gradient calculation and updates. Additionally, we introduce a dynamic completion allocation strategy to maximize GPU utilization by incorporating additional questions, further enhancing training efficiency. Experimental results demonstrate that CPPO achieves up to $8.32\times$ speedup on GSM8K and $3.51\times$ on Math while preserving or even enhancing the accuracy compared to the original GRPO. We release our code at this https URL. 

**Abstract (ZH)**: 基于组相对策略优化的完成剪枝策略优化（CPPO）以加速推理模型训练 

---
# OntoAligner: A Comprehensive Modular and Robust Python Toolkit for Ontology Alignment 

**Title (ZH)**: OntoAligner：一个全面的模块化和稳健的Python工具包用于本体对齐 

**Authors**: Hamed Babaei Giglou, Jennifer D'Souza, Oliver Karras, Sören Auer  

**Link**: [PDF](https://arxiv.org/pdf/2503.21902)  

**Abstract**: Ontology Alignment (OA) is fundamental for achieving semantic interoperability across diverse knowledge systems. We present OntoAligner, a comprehensive, modular, and robust Python toolkit for ontology alignment, designed to address current limitations with existing tools faced by practitioners. Existing tools are limited in scalability, modularity, and ease of integration with recent AI advances. OntoAligner provides a flexible architecture integrating existing lightweight OA techniques such as fuzzy matching but goes beyond by supporting contemporary methods with retrieval-augmented generation and large language models for OA. The framework prioritizes extensibility, enabling researchers to integrate custom alignment algorithms and datasets. This paper details the design principles, architecture, and implementation of the OntoAligner, demonstrating its utility through benchmarks on standard OA tasks. Our evaluation highlights OntoAligner's ability to handle large-scale ontologies efficiently with few lines of code while delivering high alignment quality. By making OntoAligner open-source, we aim to provide a resource that fosters innovation and collaboration within the OA community, empowering researchers and practitioners with a toolkit for reproducible OA research and real-world applications. 

**Abstract (ZH)**: 本体对齐（OA）是实现跨异构知识系统语义互操作性的基础。我们提出OntoAligner，这是一个全面、模块化且稳健的Python工具包，旨在解决现有工具在实践中面临的局限性。现有工具在可扩展性、模块化以及与_recent AI进展_的集成方面存在局限。OntoAligner提供了一个灵活的架构，整合了现有的轻量级本体对齐技术，如模糊匹配，但更进一步支持了与检索增强生成及大型语言模型相关的当代方法。该框架注重可扩展性，让研究人员能够集成自定义对齐算法和数据集。本文详细介绍了OntoAligner的设计原则、架构和实现，并通过标准本体对齐任务的基准测试展示了其实用性。我们的评估突显了OntoAligner能够通过少量代码高效处理大规模本体，同时保持高质量对齐的能力。通过将OntoAligner开源，我们旨在为本体对齐社区提供一个促进创新和协作的资源，使研究人员和从业者能够利用该工具包开展可再现的本体对齐研究和实际应用。 

---
# Is Best-of-N the Best of Them? Coverage, Scaling, and Optimality in Inference-Time Alignment 

**Title (ZH)**: Best-of-N是否最优？推理时序对齐的覆盖面、扩展性和最优性探究 

**Authors**: Audrey Huang, Adam Block, Qinghua Liu, Nan Jiang, Dylan J. Foster, Akshay Krishnamurthy  

**Link**: [PDF](https://arxiv.org/pdf/2503.21878)  

**Abstract**: Inference-time computation provides an important axis for scaling language model performance, but naively scaling compute through techniques like Best-of-$N$ sampling can cause performance to degrade due to reward hacking. Toward a theoretical understanding of how to best leverage additional computation, we focus on inference-time alignment which we formalize as the problem of improving a pre-trained policy's responses for a prompt of interest, given access to an imperfect reward model. We analyze the performance of inference-time alignment algorithms in terms of (i) response quality, and (ii) compute, and provide new results that highlight the importance of the pre-trained policy's coverage over high-quality responses for performance and compute scaling:
1. We show that Best-of-$N$ alignment with an ideal choice for $N$ can achieve optimal performance under stringent notions of coverage, but provably suffers from reward hacking when $N$ is large, and fails to achieve tight guarantees under more realistic coverage conditions.
2. We introduce $\texttt{InferenceTimePessimism}$, a new algorithm which mitigates reward hacking through deliberate use of inference-time compute, implementing the principle of pessimism in the face of uncertainty via rejection sampling; we prove that its performance is optimal and does not degrade with $N$, meaning it is scaling-monotonic.
We complement our theoretical results with an experimental evaluation that demonstrate the benefits of $\texttt{InferenceTimePessimism}$ across a variety of tasks and models. 

**Abstract (ZH)**: 推理时计算对于扩展语言模型性能提供了重要维度，但通过如Best-of-$N$采样等技术简单扩展计算会因奖励劫持导致性能下降。为理解和最佳利用额外计算，我们聚焦于推理时对齐问题，将其形式化为改进预训练策略对特定提示响应的问题，前提是可以访问一个不完美的奖励模型。我们从响应质量和计算量两个方面分析推理时对齐算法的性能，并提供了新的结果，强调预训练策略在高质量响应上的覆盖对于性能和计算扩展的重要性：

1. 我们证明，在严格的覆盖定义下，Best-of-$N$对齐在理想的选择$N$时可以实现最优性能，但当$N$较大时，理论上会受到奖励劫持的影响，并在更现实的覆盖条件下无法实现紧要的保障。
2. 我们引入了$\texttt{InferenceTimePessimism}$算法，通过故意利用推理时的计算来减轻奖励劫持，通过拒绝采样实现不确定性面前的悲观原则；我们证明其性能最优且不会因$N$的增加而恶化，这意味着它是计算扩展单调的。
我们通过实验证明了$\texttt{InferenceTimePessimism}$在各种任务和模型上的优势。 

---
# DSO: Aligning 3D Generators with Simulation Feedback for Physical Soundness 

**Title (ZH)**: DSO：通过模拟反馈对齐3D生成器以实现物理合理性 

**Authors**: Ruining Li, Chuanxia Zheng, Christian Rupprecht, Andrea Vedaldi  

**Link**: [PDF](https://arxiv.org/pdf/2503.22677)  

**Abstract**: Most 3D object generators focus on aesthetic quality, often neglecting physical constraints necessary in applications. One such constraint is that the 3D object should be self-supporting, i.e., remains balanced under gravity. Prior approaches to generating stable 3D objects used differentiable physics simulators to optimize geometry at test-time, which is slow, unstable, and prone to local optima. Inspired by the literature on aligning generative models to external feedback, we propose Direct Simulation Optimization (DSO), a framework to use the feedback from a (non-differentiable) simulator to increase the likelihood that the 3D generator outputs stable 3D objects directly. We construct a dataset of 3D objects labeled with a stability score obtained from the physics simulator. We can then fine-tune the 3D generator using the stability score as the alignment metric, via direct preference optimization (DPO) or direct reward optimization (DRO), a novel objective, which we introduce, to align diffusion models without requiring pairwise preferences. Our experiments show that the fine-tuned feed-forward generator, using either DPO or DRO objective, is much faster and more likely to produce stable objects than test-time optimization. Notably, the DSO framework works even without any ground-truth 3D objects for training, allowing the 3D generator to self-improve by automatically collecting simulation feedback on its own outputs. 

**Abstract (ZH)**: Most 3D对象生成器专注于美学质量，往往忽视了应用中必要的物理约束。其中之一是3D对象应具备自支撑性，即在重力作用下保持平衡。先前用于生成稳定3D对象的方法使用可微物理模拟器在测试时优化几何形状，这导致速度慢、不稳定并且容易陷入局部最优。受到将生成模型与外部反馈对齐文献的启发，我们提出了一种直接模拟优化（DSO）框架，该框架利用非可微模拟器的反馈直接增加3D生成器输出稳定3D对象的可能性。我们构建了一个带有稳定分数标签的3D对象数据集，该稳定分数由物理模拟器获得。然后，我们可以通过直接偏好优化（DPO）或直接奖励优化（DRO）——这是一种我们新提出的不需要成对偏好即可对扩散模型进行对齐的方法——使用稳定分数作为对齐度量来微调3D生成器。我们的实验表明，使用DPO或DRO目标函数的微调前馈生成器比在测试时优化生成更快速且更有可能产生稳定对象。值得注意的是，DSO框架甚至在没有真实3D对象用于训练的情况下也能有效工作，允许3D生成器通过自动收集自身输出的模拟反馈来自我改进。 

---
# Think Before Recommend: Unleashing the Latent Reasoning Power for Sequential Recommendation 

**Title (ZH)**: 深思熟虑再推荐：激发顺序推荐中的潜在推理能力 

**Authors**: Jiakai Tang, Sunhao Dai, Teng Shi, Jun Xu, Xu Chen, Wen Chen, Wu Jian, Yuning Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.22675)  

**Abstract**: Sequential Recommendation (SeqRec) aims to predict the next item by capturing sequential patterns from users' historical interactions, playing a crucial role in many real-world recommender systems. However, existing approaches predominantly adopt a direct forward computation paradigm, where the final hidden state of the sequence encoder serves as the user representation. We argue that this inference paradigm, due to its limited computational depth, struggles to model the complex evolving nature of user preferences and lacks a nuanced understanding of long-tail items, leading to suboptimal performance. To address this issue, we propose \textbf{ReaRec}, the first inference-time computing framework for recommender systems, which enhances user representations through implicit multi-step reasoning. Specifically, ReaRec autoregressively feeds the sequence's last hidden state into the sequential recommender while incorporating special reasoning position embeddings to decouple the original item encoding space from the multi-step reasoning space. Moreover, we introduce two lightweight reasoning-based learning methods, Ensemble Reasoning Learning (ERL) and Progressive Reasoning Learning (PRL), to further effectively exploit ReaRec's reasoning potential. Extensive experiments on five public real-world datasets and different SeqRec architectures demonstrate the generality and effectiveness of our proposed ReaRec. Remarkably, post-hoc analyses reveal that ReaRec significantly elevates the performance ceiling of multiple sequential recommendation backbones by approximately 30\%-50\%. Thus, we believe this work can open a new and promising avenue for future research in inference-time computing for sequential recommendation. 

**Abstract (ZH)**: 序贯推荐（SeqRec）旨在通过捕捉用户历史交互中的序贯模式来预测下一个项目，在许多实际推荐系统中发挥着重要作用。然而，现有的方法主要采用了直接前向计算范式，其中序列编码的最终隐藏状态作为用户表示。我们 argue 认为，这种推理范式由于计算深度有限，难以模拟用户偏好的复杂演变性质，并且缺乏对长尾项目的精细理解，导致性能不佳。为解决这一问题，我们提出了 ReaRec，这是第一个推荐系统的推理时计算框架，通过隐式的多步推理增强用户表示。具体而言，ReaRec 自回归地将序列的最后一个隐藏状态输入到序贯推荐器中，并结合特殊的推理位置嵌入以解耦原始项目编码空间与多步推理空间。此外，我们引入了两种轻量级的基于推理的学习方法，即集成推理学习（ERL）和逐步推理学习（PRL），以进一步有效地充分发挥 ReaRec 的推理潜力。在五个公开的实际数据集和不同 SeqRec 架构上的 extensive 实验表明，我们提出的 ReaRec 具有通用性和有效性。值得注意的是，事后分析显示，ReaRec 显著提高了多个序贯推荐基础架构的性能天花板，大约为 30%-50%。因此，我们相信这项工作可以开启未来推理时计算在序贯推荐中研究的新途径。 

---
# Exploring the Effectiveness of Multi-stage Fine-tuning for Cross-encoder Re-rankers 

**Title (ZH)**: 探索多阶段微调对交叉编码重新排ranking器有效性的影响 

**Authors**: Francesca Pezzuti, Sean MacAvaney, Nicola Tonellotto  

**Link**: [PDF](https://arxiv.org/pdf/2503.22672)  

**Abstract**: State-of-the-art cross-encoders can be fine-tuned to be highly effective in passage re-ranking. The typical fine-tuning process of cross-encoders as re-rankers requires large amounts of manually labelled data, a contrastive learning objective, and a set of heuristically sampled negatives. An alternative recent approach for fine-tuning instead involves teaching the model to mimic the rankings of a highly effective large language model using a distillation objective. These fine-tuning strategies can be applied either individually, or in sequence. In this work, we systematically investigate the effectiveness of point-wise cross-encoders when fine-tuned independently in a single stage, or sequentially in two stages. Our experiments show that the effectiveness of point-wise cross-encoders fine-tuned using contrastive learning is indeed on par with that of models fine-tuned with multi-stage approaches. Code is available for reproduction at this https URL. 

**Abstract (ZH)**: 最先进的跨编码器可以调整为在段落排序中非常有效的细调过程。跨编码器作为重新排序器的典型细调过程需要大量手动标注数据、对比学习目标以及一组启发式采样的负样本。一种最近的替代细调方法是通过蒸馏目标让模型模仿高性能大型语言模型的排序。这些细调策略可以单独应用，也可以按顺序分两个阶段应用。在本工作中，我们系统研究了单阶段独立细调以及两阶段顺序细调条件下点-wise跨编码器的有效性。实验结果显示，使用对比学习方式细调的点-wise跨编码器的效果确实与多阶段细调方法相当。相关代码可在以下网址复制：this https URL。 

---
# Challenges and Paths Towards AI for Software Engineering 

**Title (ZH)**: 面向软件工程的AI挑战与途径 

**Authors**: Alex Gu, Naman Jain, Wen-Ding Li, Manish Shetty, Yijia Shao, Ziyang Li, Diyi Yang, Kevin Ellis, Koushik Sen, Armando Solar-Lezama  

**Link**: [PDF](https://arxiv.org/pdf/2503.22625)  

**Abstract**: AI for software engineering has made remarkable progress recently, becoming a notable success within generative AI. Despite this, there are still many challenges that need to be addressed before automated software engineering reaches its full potential. It should be possible to reach high levels of automation where humans can focus on the critical decisions of what to build and how to balance difficult tradeoffs while most routine development effort is automated away. Reaching this level of automation will require substantial research and engineering efforts across academia and industry. In this paper, we aim to discuss progress towards this in a threefold manner. First, we provide a structured taxonomy of concrete tasks in AI for software engineering, emphasizing the many other tasks in software engineering beyond code generation and completion. Second, we outline several key bottlenecks that limit current approaches. Finally, we provide an opinionated list of promising research directions toward making progress on these bottlenecks, hoping to inspire future research in this rapidly maturing field. 

**Abstract (ZH)**: AI在软件工程中的应用取得了显著进展，成为生成型AI的显著成功案例。尽管如此，在实现完全自动化的软件工程之前，仍有许多挑战需要解决。应有可能实现高度自动化，使人类专注于构建的重要决策和复杂的权衡取舍，而大多数常规开发工作被自动化取代。达到这一水平的自动化将需要学术界和工业界的大量研究和工程努力。在本文中，我们旨在以三方面的方式探讨这一目标的进展。首先，我们提供了一种结构化的AI在软件工程中具体任务的分类法，强调软件工程中除了代码生成和完成之外的许多其他任务。其次，我们概述了当前方法的几个关键瓶颈。最后，我们提出了一种有倾向性的研究方向列表，旨在为解决这些瓶颈提供灵感，希望激发这一快速成熟领域的未来研究。 

---
# Generative Latent Neural PDE Solver using Flow Matching 

**Title (ZH)**: 生成型潜神经PDE求解器基于流匹配 

**Authors**: Zijie Li, Anthony Zhou, Amir Barati Farimani  

**Link**: [PDF](https://arxiv.org/pdf/2503.22600)  

**Abstract**: Autoregressive next-step prediction models have become the de-facto standard for building data-driven neural solvers to forecast time-dependent partial differential equations (PDEs). Denoise training that is closely related to diffusion probabilistic model has been shown to enhance the temporal stability of neural solvers, while its stochastic inference mechanism enables ensemble predictions and uncertainty quantification. In principle, such training involves sampling a series of discretized diffusion timesteps during both training and inference, inevitably increasing computational overhead. In addition, most diffusion models apply isotropic Gaussian noise on structured, uniform grids, limiting their adaptability to irregular domains. We propose a latent diffusion model for PDE simulation that embeds the PDE state in a lower-dimensional latent space, which significantly reduces computational costs. Our framework uses an autoencoder to map different types of meshes onto a unified structured latent grid, capturing complex geometries. By analyzing common diffusion paths, we propose to use a coarsely sampled noise schedule from flow matching for both training and testing. Numerical experiments show that the proposed model outperforms several deterministic baselines in both accuracy and long-term stability, highlighting the potential of diffusion-based approaches for robust data-driven PDE learning. 

**Abstract (ZH)**: 自回归下一步预测模型已成为构建数据驱动神经求解器以预测时间依赖偏微分方程(PDEs)的标准方法。与去噪训练紧密相关的去噪训练概率模型已被证明能够增强神经求解器的时间稳定性，其随机推断机制还能够实现 Ensemble 预测和不确定性量化。原则上，这种训练在训练和推理过程中不可避免地需要采样一系列离散化的扩散时间步长，从而增加计算开销。此外，大多数扩散模型在规则的均匀网格上应用各向同性高斯噪声，限制了它们对不规则领域域的适应性。我们提出了一种用于PDE模拟的潜在扩散模型，将PDE状态嵌入到低维潜在空间中，这显著降低了计算成本。该框架使用自编码器将不同类型的网格映射到统一的结构化潜在网格上，捕获复杂几何结构。通过分析常见的扩散路径，我们提出在训练和测试中使用从流动匹配粗采样的噪声调度。数值实验表明，所提出模型在准确性和长期稳定性方面均优于几种确定性基线，突显了基于扩散的方法在鲁棒数据驱动PDE学习中的潜力。 

---
# A Framework for Cryptographic Verifiability of End-to-End AI Pipelines 

**Title (ZH)**: 端到端人工智能管道的 cryptographic 可验证性框架 

**Authors**: Kar Balan, Robert Learney, Tim Wood  

**Link**: [PDF](https://arxiv.org/pdf/2503.22573)  

**Abstract**: The increasing integration of Artificial Intelligence across multiple industry sectors necessitates robust mechanisms for ensuring transparency, trust, and auditability of its development and deployment. This topic is particularly important in light of recent calls in various jurisdictions to introduce regulation and legislation on AI safety. In this paper, we propose a framework for complete verifiable AI pipelines, identifying key components and analyzing existing cryptographic approaches that contribute to verifiability across different stages of the AI lifecycle, from data sourcing to training, inference, and unlearning. This framework could be used to combat misinformation by providing cryptographic proofs alongside AI-generated assets to allow downstream verification of their provenance and correctness. Our findings underscore the importance of ongoing research to develop cryptographic tools that are not only efficient for isolated AI processes, but that are efficiently `linkable' across different processes within the AI pipeline, to support the development of end-to-end verifiable AI technologies. 

**Abstract (ZH)**: 跨多个行业领域的人工智能日益集成 necessitates  robust 机制以确保其开发和部署的透明性、可信度和可审计性。鉴于各司法管辖区最近对人工智能安全引入监管和立法的呼吁，该主题尤为重要。本文提出了一种完整的可验证人工智能管道框架，识别关键组件并分析贯穿人工智能生命周期各阶段的现有密码学方法，以提高不同阶段的可验证性，从数据来源到训练、推理和遗忘。该框架可通过提供与人工智能生成资产一同的密码学证明，以供下游验证其来源和正确性，从而对抗虚假信息。我们的研究结果强调了持续开发不仅适用于孤立人工智能进程的高效密码学工具的重要性，还强调了这些工具在人工智能管道内不同过程中高效“关联”的重要性，以支持端到端可验证人工智能技术的发展。 

---
# AnnoPage Dataset: Dataset of Non-Textual Elements in Documents with Fine-Grained Categorization 

**Title (ZH)**: AnnoPage 数据集：文档中非文本元素的细粒度分类数据集 

**Authors**: Martin Kišš, Michal Hradiš, Martina Dvořáková, Václav Jiroušek, Filip Kersch  

**Link**: [PDF](https://arxiv.org/pdf/2503.22526)  

**Abstract**: We introduce the AnnoPage Dataset, a novel collection of 7550 pages from historical documents, primarily in Czech and German, spanning from 1485 to the present, focusing on the late 19th and early 20th centuries. The dataset is designed to support research in document layout analysis and object detection. Each page is annotated with axis-aligned bounding boxes (AABB) representing elements of 25 categories of non-textual elements, such as images, maps, decorative elements, or charts, following the Czech Methodology of image document processing. The annotations were created by expert librarians to ensure accuracy and consistency. The dataset also incorporates pages from multiple, mainly historical, document datasets to enhance variability and maintain continuity. The dataset is divided into development and test subsets, with the test set carefully selected to maintain the category distribution. We provide baseline results using YOLO and DETR object detectors, offering a reference point for future research. The AnnoPage Dataset is publicly available on Zenodo (this https URL), along with ground-truth annotations in YOLO format. 

**Abstract (ZH)**: AnnoPage 数据集：一种包含从 1485 年至今历史文献中的 7550 页文档的新颖集合，主要使用捷克语和德语文本，重点关注 19 世纪晚期和 20 世纪早期。该数据集旨在支持文档布局分析和对象检测研究。每页都标注有表示 25 类非文本元素（如图像、地图、装饰元素或图表）的轴对齐边界框（AABB），遵循捷克图像文档处理方法学。标注工作由专家图书馆员完成，以确保准确性和一致性。数据集还整合了多个主要来自历史文献的数据集，以增强多样性和保持连贯性。数据集分为开发集和测试集，测试集精心挑选以保持类别分布的一致性。我们提供了使用 YOLO 和 DETR 对象检测器的基准结果，为未来的研究提供参考点。AnnoPage 数据集可在 Zenodo 上公开获取 (这个 https URL)，并提供 YOLO 格式的 ground-truth 注释。 

---
# Masked Self-Supervised Pre-Training for Text Recognition Transformers on Large-Scale Datasets 

**Title (ZH)**: 大规模数据集上的掩蔽自我监督预训练文本识别变换器 

**Authors**: Martin Kišš, Michal Hradiš  

**Link**: [PDF](https://arxiv.org/pdf/2503.22513)  

**Abstract**: Self-supervised learning has emerged as a powerful approach for leveraging large-scale unlabeled data to improve model performance in various domains. In this paper, we explore masked self-supervised pre-training for text recognition transformers. Specifically, we propose two modifications to the pre-training phase: progressively increasing the masking probability, and modifying the loss function to incorporate both masked and non-masked patches. We conduct extensive experiments using a dataset of 50M unlabeled text lines for pre-training and four differently sized annotated datasets for fine-tuning. Furthermore, we compare our pre-trained models against those trained with transfer learning, demonstrating the effectiveness of the self-supervised pre-training. In particular, pre-training consistently improves the character error rate of models, in some cases up to 30 % relatively. It is also on par with transfer learning but without relying on extra annotated text lines. 

**Abstract (ZH)**: 自监督学习作为一种利用大规模未标注数据提高各种领域模型性能的强大方法已经Emerged。本文我们探索文本识别变换器的掩码自监督预训练方法。具体而言，我们在预训练阶段提出了两种修改：逐步增加掩码概率，并修改损失函数以同时考虑掩码和非掩码片段。我们使用包含50百万个未标注文本行的数据集进行预训练，并使用四个不同大小的标注数据集进行微调。此外，我们还将我们的预训练模型与通过迁移学习训练的模型进行比较，以证明自监督预训练的有效性。特别是，预训练一致地提高了字符错误率，某些情况下相对提高了多达30%。另一方面，它的效果与迁移学习相当，但无需依赖额外的标注文本行。 

---
# Almost Bayesian: The Fractal Dynamics of Stochastic Gradient Descent 

**Title (ZH)**: 几乎贝叶斯：随机梯度下降的分形动态 

**Authors**: Max Hennick, Stijn De Baerdemacker  

**Link**: [PDF](https://arxiv.org/pdf/2503.22478)  

**Abstract**: We show that the behavior of stochastic gradient descent is related to Bayesian statistics by showing that SGD is effectively diffusion on a fractal landscape, where the fractal dimension can be accounted for in a purely Bayesian way. By doing this we show that SGD can be regarded as a modified Bayesian sampler which accounts for accessibility constraints induced by the fractal structure of the loss landscape. We verify our results experimentally by examining the diffusion of weights during training. These results offer insight into the factors which determine the learning process, and seemingly answer the question of how SGD and purely Bayesian sampling are related. 

**Abstract (ZH)**: 我们展示了随机梯度下降的行为与贝叶斯统计之间的关系，通过证明SGD实际上是在分形景观上的扩散过程，分形维数可以用纯粹的贝叶斯方法来解释。通过这种方式，我们表明SGD可以被视为一种修正过的贝叶斯采样器，它考虑了由损失景观分形结构引起的可访问性约束。我们通过检查训练过程中权重的扩散来实验验证这些结果。这些结果为我们提供了决定学习过程的因素提供了见解，并似乎回答了SGD和纯粹贝叶斯采样之间关系的问题。 

---
# A Causal Framework to Measure and Mitigate Non-binary Treatment Discrimination 

**Title (ZH)**: 一种用于测量和减轻非二元治疗歧视的因果框架 

**Authors**: Ayan Majumdar, Deborah D. Kanubala, Kavya Gupta, Isabel Valera  

**Link**: [PDF](https://arxiv.org/pdf/2503.22454)  

**Abstract**: Fairness studies of algorithmic decision-making systems often simplify complex decision processes, such as bail or loan approvals, into binary classification tasks. However, these approaches overlook that such decisions are not inherently binary (e.g., approve or not approve bail or loan); they also involve non-binary treatment decisions (e.g., bail conditions or loan terms) that can influence the downstream outcomes (e.g., loan repayment or reoffending). In this paper, we argue that non-binary treatment decisions are integral to the decision process and controlled by decision-makers and, therefore, should be central to fairness analyses in algorithmic decision-making. We propose a causal framework that extends fairness analyses and explicitly distinguishes between decision-subjects' covariates and the treatment decisions. This specification allows decision-makers to use our framework to (i) measure treatment disparity and its downstream effects in historical data and, using counterfactual reasoning, (ii) mitigate the impact of past unfair treatment decisions when automating decision-making. We use our framework to empirically analyze four widely used loan approval datasets to reveal potential disparity in non-binary treatment decisions and their discriminatory impact on outcomes, highlighting the need to incorporate treatment decisions in fairness assessments. Moreover, by intervening in treatment decisions, we show that our framework effectively mitigates treatment discrimination from historical data to ensure fair risk score estimation and (non-binary) decision-making processes that benefit all stakeholders. 

**Abstract (ZH)**: 算法决策系统中的公平性研究往往将复杂的决策过程（如保释或贷款审批）简化为二元分类任务。然而，这些方法忽视了这些决策本质上并非二元的（例如，批准或不批准保释或贷款），还涉及影响下游结果（如贷款偿还或重新犯罪）的非二元治疗决策。在本文中，我们主张非二元治疗决策是决策过程的重要组成部分，由决策者控制，因此在算法决策中的公平性分析中应占据中心地位。我们提出了一种因果框架，将其纳入公平性分析中，并明确区分决策主体的协变量和治疗决策。这种建模允许决策者使用我们的框架（i）在历史数据中衡量治疗差异及其下游影响，并利用反事实推理（ii）在自动化决策时减轻过去不公平治疗决策的影响。我们使用我们的框架对四个广泛使用的贷款审批数据集进行实证分析，揭示了非二元治疗决策中潜在的差异及其对结果的歧视性影响，强调了在公平评估中纳入治疗决策的必要性。此外，通过干预治疗决策，我们展示了我们的框架如何有效地从历史数据中缓解治疗歧视，以确保公平的风险评分估计和（非二元）决策过程，从而使所有相关方受益。 

---
# CoSIL: Software Issue Localization via LLM-Driven Code Repository Graph Searching 

**Title (ZH)**: CoSIL: 软件问题定位 via LLM 驱动的代码仓库图搜索 

**Authors**: Zhonghao Jiang, Xiaoxue Ren, Meng Yan, Wei Jiang, Yong Li, Zhongxin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.22424)  

**Abstract**: Large language models (LLMs) have significantly advanced autonomous software engineering, leading to a growing number of software engineering agents that assist developers in automatic program repair. Issue localization forms the basis for accurate patch generation. However, because of limitations caused by the context window length of LLMs, existing issue localization methods face challenges in balancing concise yet effective contexts and adequately comprehensive search spaces. In this paper, we introduce CoSIL, an LLM driven, simple yet powerful function level issue localization method without training or indexing. CoSIL reduces the search space through module call graphs, iteratively searches the function call graph to obtain relevant contexts, and uses context pruning to control the search direction and manage contexts effectively. Importantly, the call graph is dynamically constructed by the LLM during search, eliminating the need for pre-parsing. Experiment results demonstrate that CoSIL achieves a Top-1 localization success rate of 43 percent and 44.6 percent on SWE bench Lite and SWE bench Verified, respectively, using Qwen2.5 Coder 32B, outperforming existing methods by 8.6 to 98.2 percent. When CoSIL is applied to guide the patch generation stage, the resolved rate further improves by 9.3 to 31.5 percent. 

**Abstract (ZH)**: 大型语言模型（LLMs）显著推动了自主软件工程的发展，导致出现越来越多的软件工程代理，协助开发人员进行自动程序修复。问题定位是准确生成补丁的基础。然而，由于受限于LLMs的上下文窗口长度，现有的问题定位方法在简洁有效的情景描述和足够全面的搜索空间之间面临着平衡挑战。在本文中，我们介绍了CoSIL，这是一种由LLMs驱动、简单高效且无需训练或索引的功能级别问题定位方法。CoSIL通过模块调用图减少搜索空间，迭代搜索函数调用图以获得相关上下文，并使用上下文修剪来控制搜索方向并有效地管理上下文。重要的是，调用图在搜索过程中由LLMs动态构建，消除了预先解析的需求。实验结果表明，使用Qwen2.5 Coder 32B，CoSIL在SWE bench Lite和SWE bench Verified上的Top-1定位成功率分别为43%和44.6%，优于现有方法8.6%至98.2%。当CoSIL应用于指导补丁生成阶段时，解决率进一步提高9.3%至31.5%。 

---
# EllieSQL: Cost-Efficient Text-to-SQL with Complexity-Aware Routing 

**Title (ZH)**: EllieSQL: 基于复杂性意识路由的低成本文本到SQL转换 

**Authors**: Yizhang Zhu, Runzhi Jiang, Boyan Li, Nan Tang, Yuyu Luo  

**Link**: [PDF](https://arxiv.org/pdf/2503.22402)  

**Abstract**: Text-to-SQL automatically translates natural language queries to SQL, allowing non-technical users to retrieve data from databases without specialized SQL knowledge. Despite the success of advanced LLM-based Text-to-SQL approaches on leaderboards, their unsustainable computational costs--often overlooked--stand as the "elephant in the room" in current leaderboard-driven research, limiting their economic practicability for real-world deployment and widespread adoption. To tackle this, we exploratively propose EllieSQL, a complexity-aware routing framework that assigns queries to suitable SQL generation pipelines based on estimated complexity. We investigate multiple routers to direct simple queries to efficient approaches while reserving computationally intensive methods for complex cases. Drawing from economics, we introduce the Token Elasticity of Performance (TEP) metric, capturing cost-efficiency by quantifying the responsiveness of performance gains relative to token investment in SQL generation. Experiments show that compared to always using the most advanced methods in our study, EllieSQL with the Qwen2.5-0.5B-DPO router reduces token use by over 40% without compromising performance on Bird development set, achieving more than a 2x boost in TEP over non-routing approaches. This not only advances the pursuit of cost-efficient Text-to-SQL but also invites the community to weigh resource efficiency alongside performance, contributing to progress in sustainable Text-to-SQL. 

**Abstract (ZH)**: 复杂性意识路由框架EllieSQL：基于估计复杂性的查询路由以提升成本效率的Text-to-SQL 

---
# On-site estimation of battery electrochemical parameters via transfer learning based physics-informed neural network approach 

**Title (ZH)**: 基于物理约束神经网络的迁移学习在现场估计电池电化学参数 

**Authors**: Josu Yeregui, Iker Lopetegi, Sergio Fernandez, Erik Garayalde, Unai Iraola  

**Link**: [PDF](https://arxiv.org/pdf/2503.22396)  

**Abstract**: This paper presents a novel physical parameter estimation framework for on-site model characterization, using a two-phase modelling strategy with Physics-Informed Neural Networks (PINNs) and transfer learning (TL). In the first phase, a PINN is trained using only the physical principles of the single particle model (SPM) equations. In the second phase, the majority of the PINN parameters are frozen, while critical electrochemical parameters are set as trainable and adjusted using real-world voltage profile data. The proposed approach significantly reduces computational costs, making it suitable for real-time implementation on Battery Management Systems (BMS). Additionally, as the initial phase does not require field data, the model is easy to deploy with minimal setup requirements. With the proposed methodology, we have been able to effectively estimate relevant electrochemical parameters with operating data. This has been proved estimating diffusivities and active material volume fractions with charge data in different degradation conditions. The methodology is experimentally validated in a Raspberry Pi device using data from a standard charge profile with a 3.89\% relative accuracy estimating the active material volume fractions of a NMC cell with 82.09\% of its nominal capacity. 

**Abstract (ZH)**: 本文提出了一种用于现场模型表征的新型物理参数估计框架，采用包含物理学知识的人工神经网络（PINNs）和迁移学习（TL）的两阶段建模策略。在第一阶段，仅使用单粒子模型（SPM）方程的物理原理训练PINN。在第二阶段，固定PINN的大部分参数，将关键电化学参数设为可训练，并使用实际电压轮廓数据进行调整。所提出的方法显著降低了计算成本，使其适用于电池管理系统（BMS）的实时实现。此外，由于初始阶段不需要现场数据，该模型部署简单，无需复杂设置。通过所提出的方法，我们能够利用运行数据有效地估计相关电化学参数，证明了在不同退化条件下利用充放电数据估计扩散系数和活性材料体积分数的有效性。该方法在基于树莓派的设备上使用标准充放电数据集进行了实验验证，相对准确度为3.89%，估计镍锰钴（NMC）电池在82.09%额定容量时的活性材料体积分数。 

---
# Shapley Revisited: Tractable Responsibility Measures for Query Answers 

**Title (ZH)**: Shapley值重探：查询答案的责任度量方法 

**Authors**: Meghyn Bienvenu, Diego Figueira, Pierre Lafourcade  

**Link**: [PDF](https://arxiv.org/pdf/2503.22358)  

**Abstract**: The Shapley value, originating from cooperative game theory, has been employed to define responsibility measures that quantify the contributions of database facts to obtaining a given query answer. For non-numeric queries, this is done by considering a cooperative game whose players are the facts and whose wealth function assigns 1 or 0 to each subset of the database, depending on whether the query answer holds in the given subset. While conceptually simple, this approach suffers from a notable drawback: the problem of computing such Shapley values is #P-hard in data complexity, even for simple conjunctive queries. This motivates us to revisit the question of what constitutes a reasonable responsibility measure and to introduce a new family of responsibility measures -- weighted sums of minimal supports (WSMS) -- which satisfy intuitive properties. Interestingly, while the definition of WSMSs is simple and bears no obvious resemblance to the Shapley value formula, we prove that every WSMS measure can be equivalently seen as the Shapley value of a suitably defined cooperative game. Moreover, WSMS measures enjoy tractable data complexity for a large class of queries, including all unions of conjunctive queries. We further explore the combined complexity of WSMS computation and establish (in)tractability results for various subclasses of conjunctive queries. 

**Abstract (ZH)**: 基于 cooperaive 博弈理论的 Shapley 值已被用于定义衡量数据库事实对获得给定查询答案的贡献的责任度量。对于非数值查询，通过考虑玩家为数据库事实且财富函数对数据库子集分配 1 或 0 的博弈来进行。尽管概念上简单，但这种方法存在明显的缺点：计算此类 Shapley 值在数据复杂性上是 #P-难问题，即使是简单的合取查询也是如此。这促使我们重新审视什么是合理的责任度量，并引入一种新的责任度量家族——最小支持加权和（WSMS），它们满足直观的性质。有趣的是，尽管 WSMS 的定义简单且与 Shapley 值公式无明显联系，我们证明每个 WSMS 度量都可以等价地视为适当定义的博弈的 Shapley 值。此外，对于一类广泛的查询（包括所有合取查询的并集），WSMS 度量在数据复杂性上是可处理的。我们进一步探讨了 WSMS 计算的组合复杂性，并为多种合取查询的子类建立了可处理性和不可处理性结果。 

---
# Machine Learning Models for Soil Parameter Prediction Based on Satellite, Weather, Clay and Yield Data 

**Title (ZH)**: 基于卫星、气象、粘土和产量数据的土壤参数预测机器学习模型 

**Authors**: Calvin Kammerlander, Viola Kolb, Marinus Luegmair, Lou Scheermann, Maximilian Schmailzl, Marco Seufert, Jiayun Zhang, Denis Dalic, Torsten Schön  

**Link**: [PDF](https://arxiv.org/pdf/2503.22276)  

**Abstract**: Efficient nutrient management and precise fertilization are essential for advancing modern agriculture, particularly in regions striving to optimize crop yields sustainably. The AgroLens project endeavors to address this challenge by develop ing Machine Learning (ML)-based methodologies to predict soil nutrient levels without reliance on laboratory tests. By leveraging state of the art techniques, the project lays a foundation for acionable insights to improve agricultural productivity in resource-constrained areas, such as Africa. The approach begins with the development of a robust European model using the LUCAS Soil dataset and Sentinel-2 satellite imagery to estimate key soil properties, including phosphorus, potassium, nitrogen, and pH levels. This model is then enhanced by integrating supplementary features, such as weather data, harvest rates, and Clay AI-generated embeddings. This report details the methodological framework, data preprocessing strategies, and ML pipelines employed in this project. Advanced algorithms, including Random Forests, Extreme Gradient Boosting (XGBoost), and Fully Connected Neural Networks (FCNN), were implemented and finetuned for precise nutrient prediction. Results showcase robust model performance, with root mean square error values meeting stringent accuracy thresholds. By establishing a reproducible and scalable pipeline for soil nutrient prediction, this research paves the way for transformative agricultural applications, including precision fertilization and improved resource allocation in underresourced regions like Africa. 

**Abstract (ZH)**: 高效的养分管理与精确施肥对于推动现代农业的发展至关重要，特别是在寻求可持续优化作物产量的地区。AgroLens项目旨在通过开发基于机器学习（ML）的方法来预测土壤养分含量，从而解决这一挑战，无需依赖实验室测试。该项目借助先进的技术为基础，为资源受限地区（如非洲）提供可采取的见解，以提高农业生产效率奠定基础。该方法首先利用LUCAS土壤数据集和Sentinel-2卫星影像开发了一个稳健的欧洲模型，以估算关键土壤属性，包括磷、钾、氮和pH值。随后通过整合辅助特征，如天气数据、收获率和Clay AI生成的嵌入式特征来增强该模型。本报告详细介绍了该项目的方法论框架、数据预处理策略和所使用的ML管道。实现了包括随机森林、极端梯度提升（XGBoost）和全连接神经网络（FCNN）在内的高级算法，并对其进行了精确养分预测的优化。结果表明，模型表现出稳健的性能，均方根误差值达到了严格的准确度要求。通过建立可复制和可扩展的土壤养分预测管道，这项研究为包括精准施肥和提高资源分配效率在内的变革性农业应用铺平了道路，特别是在非洲等资源匮乏地区。 

---
# WeatherMesh-3: Fast and accurate operational global weather forecasting 

**Title (ZH)**: WeatherMesh-3：快速且准确的全球天气预报 

**Authors**: Haoxing Du, Lyna Kim, Joan Creus-Costa, Jack Michaels, Anuj Shetty, Todd Hutchinson, Christopher Riedel, John Dean  

**Link**: [PDF](https://arxiv.org/pdf/2503.22235)  

**Abstract**: We present WeatherMesh-3 (WM-3), an operational transformer-based global weather forecasting system that improves the state of the art in both accuracy and computational efficiency. We introduce the following advances: 1) a latent rollout that enables arbitrary-length predictions in latent space without intermediate encoding or decoding; and 2) a modular architecture that flexibly utilizes mixed-horizon processors and encodes multiple real-time analyses to create blended initial conditions. WM-3 generates 14-day global forecasts at 0.25-degree resolution in 12 seconds on a single RTX 4090. This represents a >100,000-fold speedup over traditional NWP approaches while achieving superior accuracy with up to 37.7% improvement in RMSE over operational models, requiring only a single consumer-grade GPU for deployment. We aim for WM-3 to democratize weather forecasting by providing an accessible, lightweight model for operational use while pushing the performance boundaries of machine learning-based weather prediction. 

**Abstract (ZH)**: 我们呈现了WeatherMesh-3 (WM-3)，这是一个基于变换器的全球天气预报系统，它在准确性和计算效率上都超越了现有最佳水平。我们引入了以下进步：1) 潜在滚动，可以在潜在空间中进行任意长度的预测，无需中间编码或解码；2) 模块化架构，灵活利用混合视窗处理器，并编码多个实时分析以生成混合初始条件。WM-3在单块RTX 4090gpu上可在12秒内生成0.25度分辨率的14天全球预报。这代表了与传统数值天气预报方法相比超过10万倍的加速，同时在均方根误差(RMSE)上最高可提高37.7%，仅需一台消费者级GPU即可部署。我们旨在通过提供一个易于访问且轻量的模型来推动气象预报的普及，同时推动基于机器学习的天气预报性能边界。 

---
# Process Reward Modeling with Entropy-Driven Uncertainty 

**Title (ZH)**: 熵驱动不确定性下的过程奖励建模 

**Authors**: Lang Cao, Renhong Chen, Yingtian Zou, Chao Peng, Wu Ning, Huacong Xu, Qian Chen, Yuxian Wang, Peishuo Su, Mofan Peng, Zijie Chen, Yitong Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.22233)  

**Abstract**: This paper presents the Entropy-Driven Unified Process Reward Model (EDU-PRM), a novel framework that approximates state-of-the-art performance in process supervision while drastically reducing training costs. EDU-PRM introduces an entropy-guided dynamic step partitioning mechanism, using logit distribution entropy to pinpoint high-uncertainty regions during token generation dynamically. This self-assessment capability enables precise step-level feedback without manual fine-grained annotation, addressing a critical challenge in process supervision. Experiments on the Qwen2.5-72B model with only 7,500 EDU-PRM-generated training queries demonstrate accuracy closely approximating the full Qwen2.5-72B-PRM (71.1% vs. 71.6%), achieving a 98% reduction in query cost compared to prior methods. This work establishes EDU-PRM as an efficient approach for scalable process reward model training. 

**Abstract (ZH)**: 熵驱动统一过程奖励模型（EDU-PRM）：一种在大幅降低训练成本的同时逼近先进性能的新框架 

---
# MFH: A Multi-faceted Heuristic Algorithm Selection Approach for Software Verification 

**Title (ZH)**: MFH：软件验证中多面向启发式算法选择方法 

**Authors**: Jie Su, Liansai Deng, Cheng Wen, Rong Wang, Zhi Ma, Nan Zhang, Cong Tian, Zhenhua Duan, Shengchao Qin  

**Link**: [PDF](https://arxiv.org/pdf/2503.22228)  

**Abstract**: Currently, many verification algorithms are available to improve the reliability of software systems. Selecting the appropriate verification algorithm typically demands domain expertise and non-trivial manpower. An automated algorithm selector is thus desired. However, existing selectors, either depend on machine-learned strategies or manually designed heuristics, encounter issues such as reliance on high-quality samples with algorithm labels and limited scalability. In this paper, an automated algorithm selection approach, namely MFH, is proposed for software verification. Our approach leverages the heuristics that verifiers producing correct results typically implement certain appropriate algorithms, and the supported algorithms by these verifiers indirectly reflect which ones are potentially applicable. Specifically, MFH embeds the code property graph (CPG) of a semantic-preserving transformed program to enhance the robustness of the prediction model. Furthermore, our approach decomposes the selection task into the sub-tasks of predicting potentially applicable algorithms and matching the most appropriate verifiers. Additionally, MFH also introduces a feedback loop on incorrect predictions to improve model prediction accuracy. We evaluate MFH on 20 verifiers and over 15,000 verification tasks. Experimental results demonstrate the effectiveness of MFH, achieving a prediction accuracy of 91.47% even without ground truth algorithm labels provided during the training phase. Moreover, the prediction accuracy decreases only by 0.84% when introducing 10 new verifiers, indicating the strong scalability of the proposed approach. 

**Abstract (ZH)**: 一种用于软件验证的自动化算法选择方法：MFH 

---
# e-person Architecture and Framework for Human-AI Co-adventure Relationship 

**Title (ZH)**: 基于e-person架构的人机共venture关系框架 

**Authors**: Kanako Esaki, Tadayuki Matsumura, Yang Shao, Hiroyuki Mizuno  

**Link**: [PDF](https://arxiv.org/pdf/2503.22181)  

**Abstract**: This paper proposes the e-person architecture for constructing a unified and incremental development of AI ethics. The e-person architecture takes the reduction of uncertainty through collaborative cognition and action with others as a unified basis for ethics. By classifying and defining uncertainty along two axes - (1) first, second, and third person perspectives, and (2) the difficulty of inference based on the depth of information - we support the development of unified and incremental development of AI ethics. In addition, we propose the e-person framework based on the free energy principle, which considers the reduction of uncertainty as a unifying principle of brain function, with the aim of implementing the e-person architecture, and we show our previous works and future challenges based on the proposed framework. 

**Abstract (ZH)**: 本文提出e-person架构以构建统一和递增的人工智能伦理。e-person架构将通过与他人的协作认知和行动减少不确定性作为伦理的基础。通过沿着两个轴分类和定义不确定性——（1）第一人、第二人和第三人视角，（2）基于信息深度的推理难度——我们支持统一和递增的人工智能伦理的发展。此外，我们基于自由能量原则提出e-person框架，将减少不确定性视为大脑功能的统一原则，旨在实施e-person架构，并基于所提出框架展示我们的先前工作和未来挑战。 

---
# AdaRank: Adaptive Rank Pruning for Enhanced Model Merging 

**Title (ZH)**: 自适应排名剪枝以增强模型合并 

**Authors**: Chanhyuk Lee, Jiho Choi, Chanryeol Lee, Donggyun Kim, Seunghoon Hong  

**Link**: [PDF](https://arxiv.org/pdf/2503.22178)  

**Abstract**: Model merging has emerged as a promising approach for unifying independently fine-tuned models into an integrated framework, significantly enhancing computational efficiency in multi-task learning. Recently, several SVD-based techniques have been introduced to exploit low-rank structures for enhanced merging, but their reliance on such manually designed rank selection often leads to cross-task interference and suboptimal performance. In this paper, we propose AdaRank, a novel model merging framework that adaptively selects the most beneficial singular directions of task vectors to merge multiple models. We empirically show that the dominant singular components of task vectors can cause critical interference with other tasks, and that naive truncation across tasks and layers degrades performance. In contrast, AdaRank dynamically prunes the singular components that cause interference and offers an optimal amount of information to each task vector by learning to prune ranks during test-time via entropy minimization. Our analysis demonstrates that such method mitigates detrimental overlaps among tasks, while empirical results show that AdaRank consistently achieves state-of-the-art performance with various backbones and number of tasks, reducing the performance gap between fine-tuned models to nearly 1%. 

**Abstract (ZH)**: 自适应奇异值选择的模型融合框架：AdaRank 

---
# When Autonomy Breaks: The Hidden Existential Risk of AI 

**Title (ZH)**: 当自主性失效：AI潜藏的 existential 风险 

**Authors**: Joshua Krook  

**Link**: [PDF](https://arxiv.org/pdf/2503.22151)  

**Abstract**: AI risks are typically framed around physical threats to humanity, a loss of control or an accidental error causing humanity's extinction. However, I argue in line with the gradual disempowerment thesis, that there is an underappreciated risk in the slow and irrevocable decline of human autonomy. As AI starts to outcompete humans in various areas of life, a tipping point will be reached where it no longer makes sense to rely on human decision-making, creativity, social care or even leadership.
What may follow is a process of gradual de-skilling, where we lose skills that we currently take for granted. Traditionally, it is argued that AI will gain human skills over time, and that these skills are innate and immutable in humans. By contrast, I argue that humans may lose such skills as critical thinking, decision-making and even social care in an AGI world. The biggest threat to humanity is therefore not that machines will become more like humans, but that humans will become more like machines. 

**Abstract (ZH)**: AI风险通常围绕着对人类的物理威胁、控制丧失或偶然错误导致人类灭绝。然而，我沿着逐步失能理论的立场argue，人类自主性的缓慢而不可逆转的下降被低估了这一风险。随着AI在生活各个领域的竞争力超越人类，将达到一个转折点，在这一点上，依赖人类决策、创造力、社会关怀或甚至领导能力就不再有意义了。随之而来的可能是技能逐步退化的过程，我们可能会失去现在习以为常的技能。传统观点认为，AI将随着时间获得人类技能，而这些技能在人类身上是天生不可变的。相反，我认为在通用人工智能的世界中，人类可能会失去批判性思维、决策能力甚至社会关怀等技能。因此，人类面临的最大威胁不是机器会变得越来越像人，而是人会变得越来越像机器。 

---
# FRASE: Structured Representations for Generalizable SPARQL Query Generation 

**Title (ZH)**: FRASE: 结构化表示以生成可泛化的SPARQL查询 

**Authors**: Papa Abdou Karim Karou Diallo, Amal Zouaq  

**Link**: [PDF](https://arxiv.org/pdf/2503.22144)  

**Abstract**: Translating natural language questions into SPARQL queries enables Knowledge Base querying for factual and up-to-date responses. However, existing datasets for this task are predominantly template-based, leading models to learn superficial mappings between question and query templates rather than developing true generalization capabilities. As a result, models struggle when encountering naturally phrased, template-free questions. This paper introduces FRASE (FRAme-based Semantic Enhancement), a novel approach that leverages Frame Semantic Role Labeling (FSRL) to address this limitation. We also present LC-QuAD 3.0, a new dataset derived from LC-QuAD 2.0, in which each question is enriched using FRASE through frame detection and the mapping of frame-elements to their argument. We evaluate the impact of this approach through extensive experiments on recent large language models (LLMs) under different fine-tuning configurations. Our results demonstrate that integrating frame-based structured representations consistently improves SPARQL generation performance, particularly in challenging generalization scenarios when test questions feature unseen templates (unknown template splits) and when they are all naturally phrased (reformulated questions). 

**Abstract (ZH)**: 基于框架语义增强的自然语言问题到SPARQL查询的转换：FRASE方法及其在LC-QuAD 3.0数据集上的应用 

---
# A Self-Supervised Learning of a Foundation Model for Analog Layout Design Automation 

**Title (ZH)**: 自监督学习为基础模型的模拟布局自动化设计 

**Authors**: Sungyu Jeong, Won Joon Choi, Junung Choi, Anik Biswas, Byungsub Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.22143)  

**Abstract**: We propose a UNet-based foundation model and its self-supervised learning method to address two key challenges: 1) lack of qualified annotated analog layout data, and 2) excessive variety in analog layout design tasks. For self-supervised learning, we propose random patch sampling and random masking techniques automatically to obtain enough training data from a small unannotated layout dataset. The obtained data are greatly augmented, less biased, equally sized, and contain enough information for excessive varieties of qualified layout patterns. By pre-training with the obtained data, the proposed foundation model can learn implicit general knowledge on layout patterns so that it can be fine-tuned for various downstream layout tasks with small task-specific datasets. Fine-tuning provides an efficient and consolidated methodology for diverse downstream tasks, reducing the enormous human effort to develop a model per task separately. In experiments, the foundation model was pre-trained using 324,000 samples obtained from 6 silicon-proved manually designed analog circuits, then it was fine-tuned for the five example downstream tasks: generating contacts, vias, dummy fingers, N-wells, and metal routings. The fine-tuned models successfully performed these tasks for more than one thousand unseen layout inputs, generating DRC/LVS-clean layouts for 96.6% of samples. Compared with training the model from scratch for the metal routing task, fine-tuning required only 1/8 of the data to achieve the same dice score of 0.95. With the same data, fine-tuning achieved a 90% lower validation loss and a 40% higher benchmark score than training from scratch. 

**Abstract (ZH)**: 基于UNet的基础模型及其自监督学习方法以应对模拟版图设计中的两个关键挑战 

---
# A Proposal for Networks Capable of Continual Learning 

**Title (ZH)**: 持续学习能力网络的提案 

**Authors**: Zeki Doruk Erden, Boi Faltings  

**Link**: [PDF](https://arxiv.org/pdf/2503.22068)  

**Abstract**: We analyze the ability of computational units to retain past responses after parameter updates, a key property for system-wide continual learning. Neural networks trained with gradient descent lack this capability, prompting us to propose Modelleyen, an alternative approach with inherent response preservation. We demonstrate through experiments on modeling the dynamics of a simple environment and on MNIST that, despite increased computational complexity and some representational limitations at its current stage, Modelleyen achieves continual learning without relying on sample replay or predefined task boundaries. 

**Abstract (ZH)**: 我们分析了计算单元在参数更新后保留过去响应的能力，这是系统级连续学习的关键属性。尽管Modelleyen目前存在一些计算复杂度增加和表示限制，但我们的实验表明，它能够在不依赖样本重放或预定义任务边界的情况下实现连续学习。 

---
# Non-Monotonic Attention-based Read/Write Policy Learning for Simultaneous Translation 

**Title (ZH)**: 基于非单调注意力的读写策略学习以实现同步翻译 

**Authors**: Zeeshan Ahmed, Frank Seide, Zhe Liu, Rastislav Rabatin, Jachym Kolar, Niko Moritz, Ruiming Xie, Simone Merello, Christian Fuegen  

**Link**: [PDF](https://arxiv.org/pdf/2503.22051)  

**Abstract**: Simultaneous or streaming machine translation generates translation while reading the input stream. These systems face a quality/latency trade-off, aiming to achieve high translation quality similar to non-streaming models with minimal latency. We propose an approach that efficiently manages this trade-off. By enhancing a pretrained non-streaming model, which was trained with a seq2seq mechanism and represents the upper bound in quality, we convert it into a streaming model by utilizing the alignment between source and target tokens. This alignment is used to learn a read/write decision boundary for reliable translation generation with minimal input. During training, the model learns the decision boundary through a read/write policy module, employing supervised learning on the alignment points (pseudo labels). The read/write policy module, a small binary classification unit, can control the quality/latency trade-off during inference. Experimental results show that our model outperforms several strong baselines and narrows the gap with the non-streaming baseline model. 

**Abstract (ZH)**: 同时或流式机器翻译在阅读输入流的同时生成翻译。这些系统面临质量和延迟之间的权衡，旨在最小延迟的情况下实现类似非流式模型的高翻译质量。我们提出了一种有效管理这种权衡的方法。通过增强一个用 seq2seq 机制预训练且代表质量上限的非流式模型，利用源和目标词之间的对齐，将其转换为流式模型。利用这种对齐来学习一个读写决策边界，以实现可靠的最小输入翻译生成。在训练过程中，模型通过读写策略模块学习决策边界，该模块在对齐点（伪标签）上使用有监督学习。读写策略模块作为一个小型二元分类单元，在推断过程中可以控制质量和延迟之间的权衡。实验结果表明，我们的模型优于多个强基准模型，并且与非流式基准模型之间的差距缩小。 

---
# Safeguarding Autonomy: a Focus on Machine Learning Decision Systems 

**Title (ZH)**: 保障自主权：关注机器学习决策系统 

**Authors**: Paula Subías-Beltrán, Oriol Pujol, Itziar de Lecuona  

**Link**: [PDF](https://arxiv.org/pdf/2503.22023)  

**Abstract**: As global discourse on AI regulation gains momentum, this paper focuses on delineating the impact of ML on autonomy and fostering awareness. Respect for autonomy is a basic principle in bioethics that establishes persons as decision-makers. While the concept of autonomy in the context of ML appears in several European normative publications, it remains a theoretical concept that has yet to be widely accepted in ML practice. Our contribution is to bridge the theoretical and practical gap by encouraging the practical application of autonomy in decision-making within ML practice by identifying the conditioning factors that currently prevent it. Consequently, we focus on the different stages of the ML pipeline to identify the potential effects on ML end-users' autonomy. To improve its practical utility, we propose a related question for each detected impact, offering guidance for identifying possible focus points to respect ML end-users autonomy in decision-making. 

**Abstract (ZH)**: 随着全球对AI监管的讨论不断升温，本文旨在阐明ML对自主性的影响，并提高对此类问题的认识。在生物伦理学中，尊重自主性是一个基本原则，确立了个人作为决策者的地位。虽然在ML背景下自主性的概念已在多个欧洲规范性出版物中出现，但这一概念在ML实践中尚未得到广泛接受。我们的贡献是通过识别目前阻碍其应用的因素，来弥合理论与实践之间的差距，鼓励在ML实践中将自主性应用于决策过程。因此，我们重点关注ML管道的不同阶段，以识别对ML终端用户自主性潜在的影响。为了提高其实用性，我们针对每个检测到的影响提出相关问题，为识别可能的重点以尊重ML终端用户在决策中的自主性提供指导。 

---
# Lobster: A GPU-Accelerated Framework for Neurosymbolic Programming 

**Title (ZH)**: lobster：一种基于GPU加速的神经符号编程框架 

**Authors**: Paul Biberstein, Ziyang Li, Joseph Devietti, Mayur Naik  

**Link**: [PDF](https://arxiv.org/pdf/2503.21937)  

**Abstract**: Neurosymbolic programs combine deep learning with symbolic reasoning to achieve better data efficiency, interpretability, and generalizability compared to standalone deep learning approaches. However, existing neurosymbolic learning frameworks implement an uneasy marriage between a highly scalable, GPU-accelerated neural component with a slower symbolic component that runs on CPUs. We propose Lobster, a unified framework for harnessing GPUs in an end-to-end manner for neurosymbolic learning. Lobster maps a general neurosymbolic language based on Datalog to the GPU programming paradigm. This mapping is implemented via compilation to a new intermediate language called APM. The extra abstraction provided by APM allows Lobster to be both flexible, supporting discrete, probabilistic, and differentiable modes of reasoning on GPU hardware with a library of provenance semirings, and performant, implementing new optimization passes. We demonstrate that Lobster programs can solve interesting problems spanning the domains of natural language processing, image processing, program reasoning, bioinformatics, and planning. On a suite of 8 applications, Lobster achieves an average speedup of 5.3x over Scallop, a state-of-the-art neurosymbolic framework, and enables scaling of neurosymbolic solutions to previously infeasible tasks. 

**Abstract (ZH)**: 神经符号程序结合深度学习与符号推理，以实现与独立深度学习方法相比更好的数据效率、可解释性和泛化能力。然而，现有的神经符号学习框架在高度可扩展的GPU加速神经组件与在CPU上运行的较慢的符号组件之间实现了不协调的结合。我们提出Lobster，一种端到端利用GPU的统一神经符号学习框架。Lobster将基于Datalog的通用神经符号语言映射到GPU编程 paradigm。这种映射通过编译到一种新的中间语言APM来实现。APM提供的额外抽象使Lobster能够灵活地支持在GPU硬件上进行离散的、概率的和可微分的推理，并且高效地实现新的优化遍历。我们展示了Lobster程序可以解决涵盖自然语言处理、图像处理、程序推理、生物信息学和规划等领域的问题。在一系列8个应用程序上，Lobster的平均加速比最先进的神经符号框架Scallop快5.3倍，并使神经符号解决方案能够缩放到此前不可行的任务。 

---
# An Efficient Training Algorithm for Models with Block-wise Sparsity 

**Title (ZH)**: 块状稀疏模型的高效训练算法 

**Authors**: Ding Zhu, Zhiqun Zuo, Mohammad Mahdi Khalili  

**Link**: [PDF](https://arxiv.org/pdf/2503.21928)  

**Abstract**: Large-scale machine learning (ML) models are increasingly being used in critical domains like education, lending, recruitment, healthcare, criminal justice, etc. However, the training, deployment, and utilization of these models demand substantial computational resources. To decrease computation and memory costs, machine learning models with sparse weight matrices are widely used in the literature. Among sparse models, those with special sparse structures (e.g., models with block-wise sparse weight matrices) fit better with the hardware accelerators and can decrease the memory and computation costs during the inference. Unfortunately, while there are several efficient training methods, none of them are designed to train a block-wise sparse model efficiently. As a result, the current methods for training block-wise sparse models start with full and dense models leading to inefficient training. In this work, we focus on training models with \textit{block-wise sparse matrices} and propose an efficient training algorithm to decrease both computation and memory costs during training and inference. In addition, we will show that our proposed method enables us to efficiently find the right block size for the sparsity pattern during the training process. Our extensive empirical and theoretical analyses show that our algorithms can decrease the computation and memory costs significantly without a performance drop compared to baselines. 

**Abstract (ZH)**: 大规模机器学习模型在教育、信贷、招聘、医疗保健、刑事司法等领域被日益广泛地应用。然而，这些模型的训练、部署和利用需要大量的计算资源。为了降低计算和内存成本，文献中广泛使用了稀疏权重矩阵的机器学习模型。在稀疏模型中，具有特殊稀疏结构的模型（例如，具有块状稀疏权重矩阵的模型）更适合硬件加速器，并能在推理过程中减少内存和计算成本。不幸的是，尽管存在多种高效的训练方法，但它们都不是为高效训练块状稀疏模型设计的。因此，当前训练块状稀疏模型的方法从全稠密模型开始，导致训练效率低下。在本文中，我们专注于训练具有块状稀疏矩阵的模型，并提出了一种高效训练算法，以在训练和推理过程中降低计算和内存成本。此外，我们将展示我们的方法使我们能够在训练过程中高效地找到合适的块大小。广泛的实验和理论分析表明，我们的算法与基线方法相比，在性能不下降的情况下，可以显著减少计算和内存成本。 

---
# Exponentially Weighted Instance-Aware Repeat Factor Sampling for Long-Tailed Object Detection Model Training in Unmanned Aerial Vehicles Surveillance Scenarios 

**Title (ZH)**: 基于无人机监控场景中长尾目标检测模型训练的指数加权实例感知重复因子采样方法 

**Authors**: Taufiq Ahmed, Abhishek Kumar, Constantino Álvarez Casado, Anlan Zhang, Tuomo Hänninen, Lauri Loven, Miguel Bordallo López, Sasu Tarkoma  

**Link**: [PDF](https://arxiv.org/pdf/2503.21893)  

**Abstract**: Object detection models often struggle with class imbalance, where rare categories appear significantly less frequently than common ones. Existing sampling-based rebalancing strategies, such as Repeat Factor Sampling (RFS) and Instance-Aware Repeat Factor Sampling (IRFS), mitigate this issue by adjusting sample frequencies based on image and instance counts. However, these methods are based on linear adjustments, which limit their effectiveness in long-tailed distributions. This work introduces Exponentially Weighted Instance-Aware Repeat Factor Sampling (E-IRFS), an extension of IRFS that applies exponential scaling to better differentiate between rare and frequent classes. E-IRFS adjusts sampling probabilities using an exponential function applied to the geometric mean of image and instance frequencies, ensuring a more adaptive rebalancing strategy. We evaluate E-IRFS on a dataset derived from the Fireman-UAV-RGBT Dataset and four additional public datasets, using YOLOv11 object detection models to identify fire, smoke, people and lakes in emergency scenarios. The results show that E-IRFS improves detection performance by 22\% over the baseline and outperforms RFS and IRFS, particularly for rare categories. The analysis also highlights that E-IRFS has a stronger effect on lightweight models with limited capacity, as these models rely more on data sampling strategies to address class imbalance. The findings demonstrate that E-IRFS improves rare object detection in resource-constrained environments, making it a suitable solution for real-time applications such as UAV-based emergency monitoring. 

**Abstract (ZH)**: Exponentially Weighted Instance-Aware Repeat Factor Sampling for Improving Rare Class Detection in Object Detection Models 

---
# LightSNN: Lightweight Architecture Search for Sparse and Accurate Spiking Neural Networks 

**Title (ZH)**: LightSNN：轻量级稀疏准确脉冲神经网络架构搜索 

**Authors**: Yesmine Abdennadher, Giovanni Perin, Riccardo Mazzieri, Jacopo Pegoraro, Michele Rossi  

**Link**: [PDF](https://arxiv.org/pdf/2503.21846)  

**Abstract**: Spiking Neural Networks (SNNs) are highly regarded for their energy efficiency, inherent activation sparsity, and suitability for real-time processing in edge devices. However, most current SNN methods adopt architectures resembling traditional artificial neural networks (ANNs), leading to suboptimal performance when applied to SNNs. While SNNs excel in energy efficiency, they have been associated with lower accuracy levels than traditional ANNs when utilizing conventional architectures. In response, in this work we present LightSNN, a rapid and efficient Neural Network Architecture Search (NAS) technique specifically tailored for SNNs that autonomously leverages the most suitable architecture, striking a good balance between accuracy and efficiency by enforcing sparsity. Based on the spiking NAS network (SNASNet) framework, a cell-based search space including backward connections is utilized to build our training-free pruning-based NAS mechanism. Our technique assesses diverse spike activation patterns across different data samples using a sparsity-aware Hamming distance fitness evaluation. Thorough experiments are conducted on both static (CIFAR10 and CIFAR100) and neuromorphic datasets (DVS128-Gesture). Our LightSNN model achieves state-of-the-art results on CIFAR10 and CIFAR100, improves performance on DVS128Gesture by 4.49%, and significantly reduces search time, most notably offering a 98x speedup over SNASNet and running 30% faster than the best existing method on DVS128Gesture. 

**Abstract (ZH)**: 基于 stabbing 的轻量级神经网络架构搜索方法（LightSNN） 

---
# ATP: Adaptive Threshold Pruning for Efficient Data Encoding in Quantum Neural Networks 

**Title (ZH)**: ATP：适配阈值剪枝在量子神经网络高效数据编码中的应用 

**Authors**: Mohamed Afane, Gabrielle Ebbrecht, Ying Wang, Juntao Chen, Junaid Farooq  

**Link**: [PDF](https://arxiv.org/pdf/2503.21815)  

**Abstract**: Quantum Neural Networks (QNNs) offer promising capabilities for complex data tasks, but are often constrained by limited qubit resources and high entanglement, which can hinder scalability and efficiency. In this paper, we introduce Adaptive Threshold Pruning (ATP), an encoding method that reduces entanglement and optimizes data complexity for efficient computations in QNNs. ATP dynamically prunes non-essential features in the data based on adaptive thresholds, effectively reducing quantum circuit requirements while preserving high performance. Extensive experiments across multiple datasets demonstrate that ATP reduces entanglement entropy and improves adversarial robustness when combined with adversarial training methods like FGSM. Our results highlight ATPs ability to balance computational efficiency and model resilience, achieving significant performance improvements with fewer resources, which will help make QNNs more feasible in practical, resource-constrained settings. 

**Abstract (ZH)**: 量子神经网络中的自适应阈值剪枝（Adaptive Threshold Pruning for Quantum Neural Networks） 

---
# Comparison of Metadata Representation Models for Knowledge Graph Embeddings 

**Title (ZH)**: 知识图嵌入中元数据表示模型的比较 

**Authors**: Shusaku Egami, Kyoumoto Matsushita, Takanori Ugai, Ken Fukuda  

**Link**: [PDF](https://arxiv.org/pdf/2503.21804)  

**Abstract**: Hyper-relational Knowledge Graphs (HRKGs) extend traditional KGs beyond binary relations, enabling the representation of contextual, provenance, and temporal information in domains, such as historical events, sensor data, video content, and narratives. HRKGs can be structured using several Metadata Representation Models (MRMs), including Reification (REF), Singleton Property (SGP), and RDF-star (RDR). However, the effects of different MRMs on KG Embedding (KGE) and Link Prediction (LP) models remain unclear. This study evaluates MRMs in the context of LP tasks, identifies the limitations of existing evaluation frameworks, and introduces a new task that ensures fair comparisons across MRMs. Furthermore, we propose a framework that effectively reflects the knowledge representations of the three MRMs in latent space. Experiments on two types of datasets reveal that REF performs well in simple HRKGs, whereas SGP is less effective. However, in complex HRKGs, the differences among MRMs in the LP tasks are minimal. Our findings contribute to an optimal knowledge representation strategy for HRKGs in LP tasks. 

**Abstract (ZH)**: 基于多个元数据表示模型的超关系知识图谱在链接预测任务中的评估与分析 

---
# Forecasting Volcanic Radiative Power (VPR) at Fuego Volcano Using Bayesian Regularized Neural Network 

**Title (ZH)**: 使用贝叶斯正则化神经网络预测危地马拉伊瓜托火山的火山辐射功率（VPR） 

**Authors**: Snehamoy Chatterjee, Greg Waite, Sidike Paheding, Luke Bowman  

**Link**: [PDF](https://arxiv.org/pdf/2503.21803)  

**Abstract**: Forecasting volcanic activity is critical for hazard assessment and risk mitigation. Volcanic Radiative Power (VPR), derived from thermal remote sensing data, serves as an essential indicator of volcanic activity. In this study, we employ Bayesian Regularized Neural Networks (BRNN) to predict future VPR values based on historical data from Fuego Volcano, comparing its performance against Scaled Conjugate Gradient (SCG) and Levenberg-Marquardt (LM) models. The results indicate that BRNN outperforms SCG and LM, achieving the lowest mean squared error (1.77E+16) and the highest R-squared value (0.50), demonstrating its superior ability to capture VPR variability while minimizing overfitting. Despite these promising results, challenges remain in improving the model's predictive accuracy. Future research should focus on integrating additional geophysical parameters, such as seismic and gas emission data, to enhance forecasting precision. The findings highlight the potential of machine learning models, particularly BRNN, in advancing volcanic activity forecasting, contributing to more effective early warning systems for volcanic hazards. 

**Abstract (ZH)**: 基于贝叶斯正则化神经网络的火山辐射功率预测及其应用： FUOGO 火山的案例研究 

---
# Efficient Joint Prediction of Multiple Future Tokens 

**Title (ZH)**: 高效联合预测多个未来词/token 

**Authors**: Kwangjun Ahn, Alex Lamb, John Langford  

**Link**: [PDF](https://arxiv.org/pdf/2503.21801)  

**Abstract**: In this short report, we introduce joint multi-token prediction (JTP), a lightweight modification of standard next-token prediction designed to enrich hidden state representations by jointly predicting multiple future tokens. Unlike previous multi-token prediction approaches, JTP strategically employs teacher forcing of future-tokens through a carefully designed representation bottleneck, allowing the model to encode rich predictive information with minimal computational overhead during training. We show that the JTP approach achieves a short-horizon belief state representation, while popular alternatives for multi-token prediction fail to do so. We demonstrate the effectiveness of our method on the synthetic star graph navigation task from from Bachmann and Nagarajan [2024], highlighting a significant performance improvement over existing methods. This manuscript presents promising preliminary results intended to stimulate further research. 

**Abstract (ZH)**: 一种轻量级的联合多令牌预测方法：实现短期信念状态表示的研究 

---
# A Novel Two-Phase Cooperative Co-evolution Framework for Large-Scale Global Optimization with Complex Overlapping 

**Title (ZH)**: 一种新型两阶段协同演化框架，用于具有复杂重叠的大规模全局优化 

**Authors**: Wenjie Qiu, Hongshu Guo, Zeyuan Ma, Yue-Jiao Gong  

**Link**: [PDF](https://arxiv.org/pdf/2503.21797)  

**Abstract**: Cooperative Co-evolution, through the decomposition of the problem space, is a primary approach for solving large-scale global optimization problems. Typically, when the subspaces are disjoint, the algorithms demonstrate significantly both effectiveness and efficiency compared to non-decomposition algorithms. However, the presence of overlapping variables complicates the decomposition process and adversely affects the performance of cooperative co-evolution. In this study, we propose a novel two-phase cooperative co-evolution framework to address large-scale global optimization problems with complex overlapping. An effective method for decomposing overlapping problems, grounded in their mathematical properties, is embedded within the framework. Additionally, a customizable benchmark for overlapping problems is introduced to extend existing benchmarks and facilitate experimentation. Extensive experiments demonstrate that the algorithm instantiated within our framework significantly outperforms existing algorithms. The results reveal the characteristics of overlapping problems and highlight the differing strengths of cooperative co-evolution and non-decomposition algorithms. Our work is open-source and accessible at: this https URL. 

**Abstract (ZH)**: 通过问题空间分解实现的合作共进化是解决大规模全局优化问题的主要方法。当子空间不相交时，该方法相较于非分解算法在有效性和效率上表现出显著优势。然而，重叠变量的存在使分解过程复杂化，并影响合作共进化的性能。在本研究中，我们提出了一种针对具有复杂重叠问题的新两阶段合作共进化框架，并基于其数学特性提出了一个有效的分解方法。此外，我们引入了一个可定制的重叠问题基准，以扩展现有基准并促进实验。广泛的实验表明，嵌入在该框架中的算法显著优于现有算法。结果揭示了重叠问题的特性，并突显了合作共进化和非分解算法的不同优势。我们的工作是开源的，可通过以下链接访问：this https URL。 

---
# Threshold Adaptation in Spiking Networks Enables Shortest Path Finding and Place Disambiguation 

**Title (ZH)**: 阈值自适应在脉冲神经网络中的实现 enables 最短路径寻找和位置歧义消解 

**Authors**: Robin Dietrich, Tobias Fischer, Nicolai Waniek, Nico Reeb, Michael Milford, Alois Knoll, Adam D. Hines  

**Link**: [PDF](https://arxiv.org/pdf/2503.21795)  

**Abstract**: Efficient spatial navigation is a hallmark of the mammalian brain, inspiring the development of neuromorphic systems that mimic biological principles. Despite progress, implementing key operations like back-tracing and handling ambiguity in bio-inspired spiking neural networks remains an open challenge. This work proposes a mechanism for activity back-tracing in arbitrary, uni-directional spiking neuron graphs. We extend the existing replay mechanism of the spiking hierarchical temporal memory (S-HTM) by our spike timing-dependent threshold adaptation (STDTA), which enables us to perform path planning in networks of spiking neurons. We further present an ambiguity dependent threshold adaptation (ADTA) for identifying places in an environment with less ambiguity, enhancing the localization estimate of an agent. Combined, these methods enable efficient identification of the shortest path to an unambiguous target. Our experiments show that a network trained on sequences reliably computes shortest paths with fewer replays than the steps required to reach the target. We further show that we can identify places with reduced ambiguity in multiple, similar environments. These contributions advance the practical application of biologically inspired sequential learning algorithms like the S-HTM towards neuromorphic localization and navigation. 

**Abstract (ZH)**: 高效的空间导航是哺乳动物大脑的一个典型特征，激发了模仿生物原理的神经形态系统的开发。尽管取得了进展，但在生物启发的脉冲神经网络中实现关键操作如回溯追踪和处理歧义仍然是一个开放的挑战。本工作提出了一种机制，用于在任意单向脉冲神经元图中执行活动回溯追踪。我们扩展了现有的脉冲层次时间记忆(S-HTM)的回放机制，引入了基于 spike 时间相关的阈值适应(STDTA)，这使我们能够在脉冲神经元网络中执行路径规划。我们进一步提出了基于歧义的阈值适应(ADTA)，用于识别环境中的歧义较少的地方，从而增强代理的定位估计。结合这两种方法，可以高效地识别到一个明确目标的最短路径。实验结果显示，一个在序列上训练的网络能够通过较少的回放可靠地计算出最短路径，直到目标。我们还展示了在多个相似环境中识别歧义较少的地方的能力。这些贡献推动了基于S-HTM等生物启发的序列学习算法的实际应用，向着神经形态定位和导航方向发展。 

---
# Architecture of Information 

**Title (ZH)**: 信息架构 

**Authors**: Yurii Parzhyn  

**Link**: [PDF](https://arxiv.org/pdf/2503.21794)  

**Abstract**: The paper explores an approach to constructing energy landscapes of a formal neuron and multilayer artificial neural networks (ANNs). Their analysis makes it possible to determine the conceptual limitations of both classification ANNs (e.g., MLP or CNN) and generative ANN models. The study of informational and thermodynamic entropy in formal neuron and ANN models leads to the conclusion about the energetic nature of informational entropy. The application of the Gibbs free energy concept allows representing the output information of ANNs as the structured part of enthalpy. Modeling ANNs as energy systems makes it possible to interpret the structure of their internal energy as an internal model of the external world, which self-organizes based on the interaction of the system's internal energy components. The control of the self-organization and evolution process of this model is carried out through an energy function (analogous to the Lyapunov function) based on reduction operators. This makes it possible to introduce a new approach to constructing self-organizing and evolutionary ANNs with direct learning, which does not require additional external algorithms. The presented research makes it possible to formulate a formal definition of information in terms of the interaction processes between the internal and external energy of the system. 

**Abstract (ZH)**: 论文探讨了构建形式神经元和多层人工神经网络（ANNs）能量景观的方法。对该模型的分析有助于确定分类ANN（如MLP或CNN）和生成性ANN模型的概念限制。通过研究形式神经元和ANN模型中的信息熵和热力学熵，得出信息熵的能量性质结论。利用吉布斯自由能的概念可以将ANN的输出信息表示为焓的有序部分。将ANN建模为能量系统，使其内部能量结构能够解释为对外部世界的内部模型，并通过系统内部能量组件的相互作用自我组织。通过对能量函数（类似于李雅普un夫函数）的控制操作来调控此模型的自我组织和进化过程。这使得可以直接学习构建自我组织和进化的ANN的新方法得以引入，无需额外的外部算法。本研究使我们能够用系统内外能之间的相互作用过程来给出信息的形式定义。 

---
# Input-Triggered Hardware Trojan Attack on Spiking Neural Networks 

**Title (ZH)**: 基于输入触发的硬件木马攻击在神经脉冲网络中 

**Authors**: Spyridon Raptis, Paul Kling, Ioannis Kaskampas, Ihsen Alouani, Haralampos-G. Stratigopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2503.21793)  

**Abstract**: Neuromorphic computing based on spiking neural networks (SNNs) is emerging as a promising alternative to traditional artificial neural networks (ANNs), offering unique advantages in terms of low power consumption. However, the security aspect of SNNs is under-explored compared to their ANN counterparts. As the increasing reliance on AI systems comes with unique security risks and challenges, understanding the vulnerabilities and threat landscape is essential as neuromorphic computing matures. In this effort, we propose a novel input-triggered Hardware Trojan (HT) attack for SNNs. The HT mechanism is condensed in the area of one neuron. The trigger mechanism is an input message crafted in the spiking domain such that a selected neuron produces a malicious spike train that is not met in normal settings. This spike train triggers a malicious modification in the neuron that forces it to saturate, firing permanently and failing to recover to its resting state even when the input activity stops. The excessive spikes pollute the network and produce misleading decisions. We propose a methodology to select an appropriate neuron and to generate the input pattern that triggers the HT payload. The attack is illustrated by simulation on three popular benchmarks in the neuromorphic community. We also propose a hardware implementation for an analog spiking neuron and a digital SNN accelerator, demonstrating that the HT has a negligible area and power footprint and, thereby, can easily evade detection. 

**Abstract (ZH)**: 基于突触神经网络（SNNs）的神经形态计算正在成为传统人工神经网络（ANNs）的有前途的替代方案，提供了独特的低功耗优势。然而，与ANNs相比，SNNs的安全性方面尚未得到充分探索。随着对人工智能系统的依赖性增加，带来了独特的安全风险和挑战，因此，在神经形态计算成熟过程中理解其脆弱性和威胁 landscape 至关重要。为此，我们提出了一种新颖的输入触发硬件 Trojan（HT）攻击方法，该方法将 HT 机制集中在单个神经元的区域。触发机制是在突触域中精心制作的输入消息，使得选定的神经元产生一种在正常情况下不会出现的恶意尖峰序列。这种尖峰序列触发了神经元的恶意修改，使其饱和，永久放电，并且即使输入活动停止也无法恢复到静息状态。过多的尖峰污染了网络并产生了误导性的决策。我们提出了一种方法来选择合适的神经元并生成触发 HT 载荷的输入模式。通过模拟在神经形态社区中流行的三个基准，展示了该攻击方法。此外，我们提出了一种模拟突触神经元和数字 SNN 加速器的硬件实现，证明了 HT 几乎没有面积和功耗开销，从而可以轻松规避检测。 

---
# March Madness Tournament Predictions Model: A Mathematical Modeling Approach 

**Title (ZH)**: March Madnesstournament预测模型：一种数学建模方法 

**Authors**: Christian McIver, Karla Avalos, Nikhil Nayak  

**Link**: [PDF](https://arxiv.org/pdf/2503.21790)  

**Abstract**: This paper proposes a model to predict the outcome of the March Madness tournament based on historical NCAA basketball data since 2013. The framework of this project is a simplification of the FiveThrityEight NCAA March Madness prediction model, where the only four predictors of interest are Adjusted Offensive Efficiency (ADJOE), Adjusted Defensive Efficiency (ADJDE), Power Rating, and Two-Point Shooting Percentage Allowed. A logistic regression was utilized with the aforementioned metrics to generate a probability of a particular team winning each game. Then, a tournament simulation is developed and compared to real-world March Madness brackets to determine the accuracy of the model. Accuracies of performance were calculated using a naive approach and a Spearman rank correlation coefficient. 

**Abstract (ZH)**: 基于2013年以来NCAA历史篮球数据的March Madnesstournament结果预测模型 

---
