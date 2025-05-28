# Vision-Based Risk Aware Emergency Landing for UAVs in Complex Urban Environments 

**Title (ZH)**: 基于视觉的风险感知紧急着陆方法研究：复杂城市环境中的无人机应用 

**Authors**: Julio de la Torre-Vanegas, Miguel Soriano-Garcia, Israel Becerra, Diego Mercado-Ravell  

**Link**: [PDF](https://arxiv.org/pdf/2505.20423)  

**Abstract**: Landing safely in crowded urban environments remains an essential yet challenging endeavor for Unmanned Aerial Vehicles (UAVs), especially in emergency situations. In this work, we propose a risk-aware approach that harnesses semantic segmentation to continuously evaluate potential hazards in the drone's field of view. By using a specialized deep neural network to assign pixel-level risk values and applying an algorithm based on risk maps, our method adaptively identifies a stable Safe Landing Zone (SLZ) despite moving critical obstacles such as vehicles, people, etc., and other visual challenges like shifting illumination. A control system then guides the UAV toward this low-risk region, employing altitude-dependent safety thresholds and temporal landing point stabilization to ensure robust descent trajectories. Experimental validation in diverse urban environments demonstrates the effectiveness of our approach, achieving over 90% landing success rates in very challenging real scenarios, showing significant improvements in various risk metrics. Our findings suggest that risk-oriented vision methods can effectively help reduce the risk of accidents in emergency landing situations, particularly in complex, unstructured, urban scenarios, densely populated with moving risky obstacles, while potentiating the true capabilities of UAVs in complex urban operations. 

**Abstract (ZH)**: 在拥挤的城市环境中安全着陆仍然是无人机（UAV）的一项必不可少但极具挑战的任务，尤其是在紧急情况下。本文提出一种风险感知的方法，利用语义分割持续评估无人机视野中的潜在危险。通过使用专门的深神经网络为像素级分配风险值，并基于风险图的应用算法，我们的方法能够在移动的障碍物（如车辆、人员等）和其他视觉挑战（如光照变化）下自适应地识别一个稳定的安全着陆区（SLZ）。随后的控制系统引导无人机向低风险区域降落，采用高度依赖的安全阈值和时间窗口内的着陆点稳定技术，以确保稳健的下降轨迹。在多样化的城市环境中进行的实验验证表明，该方法的有效性，在极其挑战的真实场景中实现了超过90%的着陆成功率，并在多种风险指标上显示出显著改进。我们的研究结果表明，风险导向的视觉方法能够有效帮助减少紧急着陆情况下的事故风险，特别是在复杂的、非结构化的、充满移动危险障碍物的城市场景中，同时增强无人机在复杂城市操作中的真正能力。 

---
# Structure from Collision 

**Title (ZH)**: 碰撞结构重建 

**Authors**: Takuhiro Kaneko  

**Link**: [PDF](https://arxiv.org/pdf/2505.21335)  

**Abstract**: Recent advancements in neural 3D representations, such as neural radiance fields (NeRF) and 3D Gaussian splatting (3DGS), have enabled the accurate estimation of 3D structures from multiview images. However, this capability is limited to estimating the visible external structure, and identifying the invisible internal structure hidden behind the surface is difficult. To overcome this limitation, we address a new task called Structure from Collision (SfC), which aims to estimate the structure (including the invisible internal structure) of an object from appearance changes during collision. To solve this problem, we propose a novel model called SfC-NeRF that optimizes the invisible internal structure of an object through a video sequence under physical, appearance (i.e., visible external structure)-preserving, and keyframe constraints. In particular, to avoid falling into undesirable local optima owing to its ill-posed nature, we propose volume annealing; that is, searching for global optima by repeatedly reducing and expanding the volume. Extensive experiments on 115 objects involving diverse structures (i.e., various cavity shapes, locations, and sizes) and material properties revealed the properties of SfC and demonstrated the effectiveness of the proposed SfC-NeRF. 

**Abstract (ZH)**: Recent advancements in神经网络三维表示，如神经辐射场（NeRF）和三维高斯散点（3DGS），使从多视角图像准确估计三维结构成为可能。然而，这种能力仅限于估计可见的外部结构，而识别隐藏在表面背后的不可见内部结构则难以实现。为克服这一限制，我们提出了一项新的任务，即碰撞结构恢复（Structure from Collision，SfC），该任务旨在通过碰撞期间的外观变化估计物体的结构（包括不可见的内部结构）。为了解决这一问题，我们提出了一种新型模型SfC-NeRF，该模型在保持物理约束、外观（即可见的外部结构）约束和关键帧约束的情况下，优化物体的不可见内部结构。特别地，为了避免由于其病态性质而陷入不良的局部极值，我们提出了一种体积退火方法，通过反复减小和扩大体积来搜索全局极值。针对115个具有多样结构（即各种空腔形状、位置和大小）和材料性质的物体进行的广泛实验揭示了SfC的性质，并证明了所提出的SfC-NeRF的有效性。 

---
# RefAV: Towards Planning-Centric Scenario Mining 

**Title (ZH)**: 面向规划中心的情景挖掘 

**Authors**: Cainan Davidson, Deva Ramanan, Neehar Peri  

**Link**: [PDF](https://arxiv.org/pdf/2505.20981)  

**Abstract**: Autonomous Vehicles (AVs) collect and pseudo-label terabytes of multi-modal data localized to HD maps during normal fleet testing. However, identifying interesting and safety-critical scenarios from uncurated driving logs remains a significant challenge. Traditional scenario mining techniques are error-prone and prohibitively time-consuming, often relying on hand-crafted structured queries. In this work, we revisit spatio-temporal scenario mining through the lens of recent vision-language models (VLMs) to detect whether a described scenario occurs in a driving log and, if so, precisely localize it in both time and space. To address this problem, we introduce RefAV, a large-scale dataset of 10,000 diverse natural language queries that describe complex multi-agent interactions relevant to motion planning derived from 1000 driving logs in the Argoverse 2 Sensor dataset. We evaluate several referential multi-object trackers and present an empirical analysis of our baselines. Notably, we find that naively repurposing off-the-shelf VLMs yields poor performance, suggesting that scenario mining presents unique challenges. Our code and dataset are available at this https URL and this https URL 

**Abstract (ZH)**: 自主车辆（AVs）在常规车队测试过程中收集并伪标签了大量的多模态数据，这些数据与高清地图相关联。然而，从未经整理的驾驶日志中识别出有趣且安全关键的场景仍然是一项重大挑战。传统的场景挖掘技术容易出错且耗时过长，往往依赖于手工艺品结构化查询。在本文中，我们通过近期的视觉-语言模型（VLMs）的观点回顾时空场景挖掘，以检测描述的场景是否出现在驾驶日志中，如果出现，则精确地在时间和空间上定位。为了解决这个问题，我们引入了RefAV，这是一个包含10,000个多样化的自然语言查询的大规模数据集，这些查询描述了从Argoverse 2 Sensor数据集中1000个驾驶日志中获得的与运动规划相关的复杂多智能体交互。我们评估了几种引用型多对象跟踪器，并对我们的基线进行了经验分析。值得注意的是，我们发现简单地重新利用现成的VLMs性能很差，这表明场景挖掘提出了独特的挑战。我们的代码和数据集可以在以下链接获取：this https URL 和 this https URL。 

---
# OmniIndoor3D: Comprehensive Indoor 3D Reconstruction 

**Title (ZH)**: 全方位室内3D重建 

**Authors**: Xiaobao Wei, Xiaoan Zhang, Hao Wang, Qingpo Wuwu, Ming Lu, Wenzhao Zheng, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20610)  

**Abstract**: We propose a novel framework for comprehensive indoor 3D reconstruction using Gaussian representations, called OmniIndoor3D. This framework enables accurate appearance, geometry, and panoptic reconstruction of diverse indoor scenes captured by a consumer-level RGB-D camera. Since 3DGS is primarily optimized for photorealistic rendering, it lacks the precise geometry critical for high-quality panoptic reconstruction. Therefore, OmniIndoor3D first combines multiple RGB-D images to create a coarse 3D reconstruction, which is then used to initialize the 3D Gaussians and guide the 3DGS training. To decouple the optimization conflict between appearance and geometry, we introduce a lightweight MLP that adjusts the geometric properties of 3D Gaussians. The introduced lightweight MLP serves as a low-pass filter for geometry reconstruction and significantly reduces noise in indoor scenes. To improve the distribution of Gaussian primitives, we propose a densification strategy guided by panoptic priors to encourage smoothness on planar surfaces. Through the joint optimization of appearance, geometry, and panoptic reconstruction, OmniIndoor3D provides comprehensive 3D indoor scene understanding, which facilitates accurate and robust robotic navigation. We perform thorough evaluations across multiple datasets, and OmniIndoor3D achieves state-of-the-art results in appearance, geometry, and panoptic reconstruction. We believe our work bridges a critical gap in indoor 3D reconstruction. The code will be released at: this https URL 

**Abstract (ZH)**: OmniIndoor3D：基于高斯表示的综合室内三维重建框架 

---
# WeatherEdit: Controllable Weather Editing with 4D Gaussian Field 

**Title (ZH)**: WeatherEdit: 基于4D高斯场的可控天气编辑 

**Authors**: Chenghao Qian, Wenjing Li, Yuhu Guo, Gustav Markkula  

**Link**: [PDF](https://arxiv.org/pdf/2505.20471)  

**Abstract**: In this work, we present WeatherEdit, a novel weather editing pipeline for generating realistic weather effects with controllable types and severity in 3D scenes. Our approach is structured into two key components: weather background editing and weather particle construction. For weather background editing, we introduce an all-in-one adapter that integrates multiple weather styles into a single pretrained diffusion model, enabling the generation of diverse weather effects in 2D image backgrounds. During inference, we design a Temporal-View (TV-) attention mechanism that follows a specific order to aggregate temporal and spatial information, ensuring consistent editing across multi-frame and multi-view images. To construct the weather particles, we first reconstruct a 3D scene using the edited images and then introduce a dynamic 4D Gaussian field to generate snowflakes, raindrops and fog in the scene. The attributes and dynamics of these particles are precisely controlled through physical-based modelling and simulation, ensuring realistic weather representation and flexible severity adjustments. Finally, we integrate the 4D Gaussian field with the 3D scene to render consistent and highly realistic weather effects. Experiments on multiple driving datasets demonstrate that WeatherEdit can generate diverse weather effects with controllable condition severity, highlighting its potential for autonomous driving simulation in adverse weather. See project page: this https URL 

**Abstract (ZH)**: 基于可控类型和严重程度生成现实天气效果的新型天气编辑管道WeatherEdit 

---
# RetroMotion: Retrocausal Motion Forecasting Models are Instructable 

**Title (ZH)**: RetroMotion: 反向因果运动预测模型可受控 víctima 

**Authors**: Royden Wagner, Omer Sahin Tas, Felix Hauser, Marlon Steiner, Dominik Strutz, Abhishek Vivekanandan, Carlos Fernandez, Christoph Stiller  

**Link**: [PDF](https://arxiv.org/pdf/2505.20414)  

**Abstract**: Motion forecasts of road users (i.e., agents) vary in complexity as a function of scene constraints and interactive behavior. We address this with a multi-task learning method for motion forecasting that includes a retrocausal flow of information. The corresponding tasks are to forecast (1) marginal trajectory distributions for all modeled agents and (2) joint trajectory distributions for interacting agents. Using a transformer model, we generate the joint distributions by re-encoding marginal distributions followed by pairwise modeling. This incorporates a retrocausal flow of information from later points in marginal trajectories to earlier points in joint trajectories. Per trajectory point, we model positional uncertainty using compressed exponential power distributions. Notably, our method achieves state-of-the-art results in the Waymo Interaction Prediction dataset and generalizes well to the Argoverse 2 dataset. Additionally, our method provides an interface for issuing instructions through trajectory modifications. Our experiments show that regular training of motion forecasting leads to the ability to follow goal-based instructions and to adapt basic directional instructions to the scene context. Code: this https URL 

**Abstract (ZH)**: 基于多任务学习的考虑回因效应的运动预测方法：预测道路使用者的运动轨迹在不同场景约束和交互行为下的复杂性变化，并通过变换器模型生成联合轨迹分布。该方法包含轨迹点位置不确定性建模，并在Waymo Interaction Prediction数据集和Argoverse 2数据集上取得了优异结果且具有良好泛化能力，还提供了通过轨迹修改发布指令的接口。实验表明，常规训练能实现基于目标的指令跟随和基本方向指令的场景适应。代码：https://this.url 

---
# Jigsaw-Puzzles: From Seeing to Understanding to Reasoning in Vision-Language Models 

**Title (ZH)**: 拼图游戏：从视觉感知到理解再到推理在 Vision-Language 模型中的应用 

**Authors**: Zesen Lyu, Dandan Zhang, Wei Ye, Fangdi Li, Zhihang Jiang, Yao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20728)  

**Abstract**: Spatial reasoning is a core component of human cognition, enabling individuals to perceive, comprehend, and interact with the physical world. It relies on a nuanced understanding of spatial structures and inter-object relationships, serving as the foundation for complex reasoning and decision-making. To investigate whether current vision-language models (VLMs) exhibit similar capability, we introduce Jigsaw-Puzzles, a novel benchmark consisting of 1,100 carefully curated real-world images with high spatial complexity. Based on this dataset, we design five tasks to rigorously evaluate VLMs' spatial perception, structural understanding, and reasoning capabilities, while deliberately minimizing reliance on domain-specific knowledge to better isolate and assess the general spatial reasoning capability. We conduct a comprehensive evaluation across 24 state-of-the-art VLMs. The results show that even the strongest model, Gemini-2.5-Pro, achieves only 77.14% overall accuracy and performs particularly poorly on the Order Generation task, with only 30.00% accuracy, far below the performance exceeding 90% achieved by human participants. This persistent gap underscores the need for continued progress, positioning Jigsaw-Puzzles as a challenging and diagnostic benchmark for advancing spatial reasoning research in VLMs. 

**Abstract (ZH)**: 空间推理是人类认知的核心组件，使个体能够感知、理解和与物理世界互动。它依赖于对空间结构和物体间关系的细腻理解，为复杂推理和决策奠定基础。为了探究当前视觉-语言模型（VLMs）是否具备类似的能力，我们引入了一种新的基准Jigsaw-Puzzles，该基准包含1100张精心挑选的实际场景图像，具有高度的空间复杂性。基于这个数据集，我们设计了五个任务以严格评估VLMs的空间感知、结构理解和推理能力，同时尽量减少对领域特定知识的依赖，以更好地隔离和评估其通用空间推理能力。我们在24种最新的VLMs中进行了全面评估。结果显示，即使是最强的模型Gemini-2.5-Pro，总体准确率也只有77.14%，在序列生成任务上的准确率仅为30.00%，远低于人类参与者超过90%的表现。这一持续的差距突显了继续进步的必要性，将Jigsaw-Puzzles定位为促进VLMs空间推理研究的一个挑战性且诊断性的基准。 

---
# Reconceptualizing Smart Microscopy: From Data Collection to Knowledge Creation by Multi-Agent Integration 

**Title (ZH)**: 重新构想智能显微镜：从数据采集到知识创造的多代理集成方法 

**Authors**: P.S. Kesavan, Pontus Nordenfelt  

**Link**: [PDF](https://arxiv.org/pdf/2505.20466)  

**Abstract**: Smart microscopy represents a paradigm shift in biological imaging, moving from passive observation tools to active collaborators in scientific inquiry. Enabled by advances in automation, computational power, and artificial intelligence, these systems are now capable of adaptive decision-making and real-time experimental control. Here, we introduce a theoretical framework that reconceptualizes smart microscopy as a partner in scientific investigation. Central to our framework is the concept of the 'epistemic-empirical divide' in cellular investigation-the gap between what is observable (empirical domain) and what must be understood (epistemic domain). We propose six core design principles: epistemic-empirical awareness, hierarchical context integration, an evolution from detection to perception, adaptive measurement frameworks, narrative synthesis capabilities, and cross-contextual reasoning. Together, these principles guide a multi-agent architecture designed to align empirical observation with the goals of scientific understanding. Our framework provides a roadmap for building microscopy systems that go beyond automation to actively support hypothesis generation, insight discovery, and theory development, redefining the role of scientific instruments in the process of knowledge creation. 

**Abstract (ZH)**: 智能显微镜代表了生物成像领域的一场范式转变，从被动观察工具转变为科学探究的主动合作者。得益于自动化、计算能力和人工智能的进步，这些系统现在能够进行适应性决策和实时实验控制。在这里，我们提出了一种理论框架，重新构想了智能显微镜作为科学研究的合作伙伴。我们框架的核心是细胞研究中“知识与经验的鸿沟”的概念——可观察到的领域与需要理解的领域之间的差距。我们建议六个核心设计原则：知识与经验的意识、层级上下文整合、从检测到感知的转变、适应性测量框架、叙述性综合能力和跨上下文推理。这些原则指导一个多代理架构的设计，旨在使经验观察与科学研究的目标相一致。我们的框架提供了一条道路，用于构建超越自动化、积极支持假说生成、洞察发现和理论发展的显微镜系统，重新定义了科学仪器在知识创造过程中的角色。 

---
# Be Decisive: Noise-Induced Layouts for Multi-Subject Generation 

**Title (ZH)**: 果断行动：噪声诱导的多主题生成布局 

**Authors**: Omer Dahary, Yehonathan Cohen, Or Patashnik, Kfir Aberman, Daniel Cohen-Or  

**Link**: [PDF](https://arxiv.org/pdf/2505.21488)  

**Abstract**: Generating multiple distinct subjects remains a challenge for existing text-to-image diffusion models. Complex prompts often lead to subject leakage, causing inaccuracies in quantities, attributes, and visual features. Preventing leakage among subjects necessitates knowledge of each subject's spatial location. Recent methods provide these spatial locations via an external layout control. However, enforcing such a prescribed layout often conflicts with the innate layout dictated by the sampled initial noise, leading to misalignment with the model's prior. In this work, we introduce a new approach that predicts a spatial layout aligned with the prompt, derived from the initial noise, and refines it throughout the denoising process. By relying on this noise-induced layout, we avoid conflicts with externally imposed layouts and better preserve the model's prior. Our method employs a small neural network to predict and refine the evolving noise-induced layout at each denoising step, ensuring clear boundaries between subjects while maintaining consistency. Experimental results show that this noise-aligned strategy achieves improved text-image alignment and more stable multi-subject generation compared to existing layout-guided techniques, while preserving the rich diversity of the model's original distribution. 

**Abstract (ZH)**: 现有的文本到图像扩散模型在生成多个独立主题方面仍面临挑战。复杂的提示 often lead to subject leakage, causing inaccuracies in quantities, attributes, and visual features. 预防主题之间的泄漏需要了解每个主题的空间位置。最近的方法通过外部布局控制提供这些空间位置。然而，强制这种预定布局往往会与采样初始噪声所决定的固有布局发生冲突，导致与模型先验的对齐偏差。在本文中，我们提出了一种新方法，该方法预测与提示、初始噪声相一致的空间布局，并在整个去噪过程中对其进行 refinement。通过依赖这种噪声诱导的布局，我们避免了外部强加布局的冲突，更好地保留了模型的先验。该方法使用一个小型神经网络，在每次去噪步骤中预测并细化 evolving noise-induced layout，确保各主题之间的清晰边界，同时保持一致性。实验结果表明，这种噪声对齐策略在文本图像对齐和多主题生成的稳定性方面优于现有布局引导技术，同时保留了模型原始分布的丰富多样性。 

---
# LazyVLM: Neuro-Symbolic Approach to Video Analytics 

**Title (ZH)**: LazyVLM: 神经符号视频分析方法 

**Authors**: Xiangru Jian, Wei Pang, Zhengyuan Dong, Chao Zhang, M. Tamer Özsu  

**Link**: [PDF](https://arxiv.org/pdf/2505.21459)  

**Abstract**: Current video analytics approaches face a fundamental trade-off between flexibility and efficiency. End-to-end Vision Language Models (VLMs) often struggle with long-context processing and incur high computational costs, while neural-symbolic methods depend heavily on manual labeling and rigid rule design. In this paper, we introduce LazyVLM, a neuro-symbolic video analytics system that provides a user-friendly query interface similar to VLMs, while addressing their scalability limitation. LazyVLM enables users to effortlessly drop in video data and specify complex multi-frame video queries using a semi-structured text interface for video analytics. To address the scalability limitations of VLMs, LazyVLM decomposes multi-frame video queries into fine-grained operations and offloads the bulk of the processing to efficient relational query execution and vector similarity search. We demonstrate that LazyVLM provides a robust, efficient, and user-friendly solution for querying open-domain video data at scale. 

**Abstract (ZH)**: 当前的视频分析方法在灵活性和效率之间存在根本性的权衡。端到端的视觉语言模型（VLMs）常常难以处理长序列处理，并产生较高的计算成本，而神经符号方法则依赖于手动标注和严格的规则设计。本文介绍了LazyVLM，这是一种神经符号视频分析系统，提供了类似于VLMs的用户友好查询接口，同时解决了它们的可扩展性限制。LazyVLM使用户能够轻松地导入视频数据，并使用半结构化文本接口指定复杂的多帧视频查询。为了应对VLMs的可扩展性限制，LazyVLM将多帧视频查询分解为细粒度操作，并将大部分处理工作卸载到高效的关联查询执行和向量相似性搜索。我们证明，LazyVLM提供了一种 robust、高效且用户友好的解决方案，用于在大规模查询开放域视频数据。 

---
# LPOI: Listwise Preference Optimization for Vision Language Models 

**Title (ZH)**: 列级偏好优化for vision-language模型 

**Authors**: Fatemeh Pesaran Zadeh, Yoojin Oh, Gunhee Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.21061)  

**Abstract**: Aligning large VLMs with human preferences is a challenging task, as methods like RLHF and DPO often overfit to textual information or exacerbate hallucinations. Although augmenting negative image samples partially addresses these pitfalls, no prior work has employed listwise preference optimization for VLMs, due to the complexity and cost of constructing listwise image samples. In this work, we propose LPOI, the first object-aware listwise preference optimization developed for reducing hallucinations in VLMs. LPOI identifies and masks a critical object in the image, and then interpolates the masked region between the positive and negative images to form a sequence of incrementally more complete images. The model is trained to rank these images in ascending order of object visibility, effectively reducing hallucinations while retaining visual fidelity. LPOI requires no extra annotations beyond standard pairwise preference data, as it automatically constructs the ranked lists through object masking and interpolation. Comprehensive experiments on MMHalBench, AMBER, and Object HalBench confirm that LPOI outperforms existing preference optimization methods in reducing hallucinations and enhancing VLM performance. We make the code available at this https URL. 

**Abstract (ZH)**: 将大规模VLM与人类偏好对齐是一项具有挑战性的任务，因为方法如RLHF和DPO往往过度拟合到文本信息或加剧幻觉现象。尽管部分通过扩充负样本图像可以解决这些问题，但先前工作未采用列表级偏好优化方法，主要是因为构建列表级图像样本的复杂性和成本。在本文中，我们提出LPOI，这是一种用于减少VLM幻觉现象的第一种对象感知列表级偏好优化方法。LPOI识别并屏蔽图像中的关键对象，然后在正图像和负图像之间插值屏蔽区域，形成逐渐更完整的图像序列。模型被训练为按对象可视度递增顺序对这些图像进行-ranking，从而有效减少幻觉现象同时保留视觉保真度。LPOI不需要额外的标注，因为它可以通过对象屏蔽和插值自动构建排名列表。在MMHalBench、AMBER和Object HalBench上的全面实验表明，LPOI在减少幻觉现象和增强VLM性能方面优于现有偏好优化方法。代码已发布在此HTTPS URL。 

---
# RainFusion: Adaptive Video Generation Acceleration via Multi-Dimensional Visual Redundancy 

**Title (ZH)**: RainFusion：通过多维度视觉冗余实现的自适应视频生成加速 

**Authors**: Aiyue Chen, Bin Dong, Jingru Li, Jing Lin, Yiwu Yao, Gongyi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.21036)  

**Abstract**: Video generation using diffusion models is highly computationally intensive, with 3D attention in Diffusion Transformer (DiT) models accounting for over 80\% of the total computational resources. In this work, we introduce {\bf RainFusion}, a novel training-free sparse attention method that exploits inherent sparsity nature in visual data to accelerate attention computation while preserving video quality. Specifically, we identify three unique sparse patterns in video generation attention calculations--Spatial Pattern, Temporal Pattern and Textural Pattern. The sparse pattern for each attention head is determined online with negligible overhead (\textasciitilde\,0.2\%) with our proposed {\bf ARM} (Adaptive Recognition Module) during inference. Our proposed {\bf RainFusion} is a plug-and-play method, that can be seamlessly integrated into state-of-the-art 3D-attention video generation models without additional training or calibration. We evaluate our method on leading open-sourced models including HunyuanVideo, OpenSoraPlan-1.2 and CogVideoX-5B, demonstrating its broad applicability and effectiveness. Experimental results show that RainFusion achieves over {\bf 2\(\times\)} speedup in attention computation while maintaining video quality, with only a minimal impact on VBench scores (-0.2\%). 

**Abstract (ZH)**: 基于扩散模型的视频生成 highly 计算密集，其中 3D 注意力在 Diffusion Transformer (DiT) 模型中占用了超过 80% 的总计算资源。本文提出了一种名为 RainFusion 的新型无训练稀疏注意力方法，该方法通过利用视觉数据的固有稀疏性来加速注意力计算，同时保持视频质量。具体地，我们在推理过程中使用提出的 ARM （自适应识别模块）在线确定每个注意力头的稀疏模式，计算开销约为 0.2%。RainFusion 是一种即插即用的方法，可以无缝集成到最先进的 3D 注意力视频生成模型中，无需额外的训练或校准。我们在领先的开源模型 HunyuanVideo、OpenSoraPlan-1.2 和 CogVideoX-5B 上评估了该方法，展示了其广泛适用性和有效性。实验结果表明，RainFusion 在保持视频质量的同时，注意力计算速度提高了超过 2 倍，仅对 VBench 分数产生轻微影响（-0.2%）。 

---
# FeatInv: Spatially resolved mapping from feature space to input space using conditional diffusion models 

**Title (ZH)**: FeatInv：使用条件扩散模型从特征空间到输入空间的空域解析映射 

**Authors**: Nils Neukirch, Johanna Vielhaben, Nils Strodthoff  

**Link**: [PDF](https://arxiv.org/pdf/2505.21032)  

**Abstract**: Internal representations are crucial for understanding deep neural networks, such as their properties and reasoning patterns, but remain difficult to interpret. While mapping from feature space to input space aids in interpreting the former, existing approaches often rely on crude approximations. We propose using a conditional diffusion model - a pretrained high-fidelity diffusion model conditioned on spatially resolved feature maps - to learn such a mapping in a probabilistic manner. We demonstrate the feasibility of this approach across various pretrained image classifiers from CNNs to ViTs, showing excellent reconstruction capabilities. Through qualitative comparisons and robustness analysis, we validate our method and showcase possible applications, such as the visualization of concept steering in input space or investigations of the composite nature of the feature space. This approach has broad potential for improving feature space understanding in computer vision models. 

**Abstract (ZH)**: 内部表征对于理解深度神经网络至关重要，包括其属性和推理模式，但仍然难以解释。虽然将特征空间映射到输入空间有助于解释前者，但现有方法往往依赖于粗略的近似。我们提出使用条件扩散模型——一种基于空间解析特征图预训练的高保真扩散模型——以概率方式学习这种映射。我们展示了该方法在各种从CNN到ViT的预训练图像分类器中的可行性，显示出出色重构能力。通过定性比较和鲁棒性分析，我们验证了该方法并展示了其潜在应用，例如输入空间的概念导向可视化或特征空间复合性质的研究。该方法在计算机视觉模型的特征空间理解方面具有广泛的潜在应用价值。 

---
# A Stereotype Content Analysis on Color-related Social Bias in Large Vision Language Models 

**Title (ZH)**: 颜色相关社会偏见在大型视觉语言模型中的刻板印象内容分析 

**Authors**: Junhyuk Choi, Minju Kim, Yeseon Hong, Bugeun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.20901)  

**Abstract**: As large vision language models(LVLMs) rapidly advance, concerns about their potential to learn and generate social biases and stereotypes are increasing. Previous studies on LVLM's stereotypes face two primary limitations: metrics that overlooked the importance of content words, and datasets that overlooked the effect of color. To address these limitations, this study introduces new evaluation metrics based on the Stereotype Content Model (SCM). We also propose BASIC, a benchmark for assessing gender, race, and color stereotypes. Using SCM metrics and BASIC, we conduct a study with eight LVLMs to discover stereotypes. As a result, we found three findings. (1) The SCM-based evaluation is effective in capturing stereotypes. (2) LVLMs exhibit color stereotypes in the output along with gender and race ones. (3) Interaction between model architecture and parameter sizes seems to affect stereotypes. We release BASIC publicly on [anonymized for review]. 

**Abstract (ZH)**: 随着大型视觉语言模型(LVLMs)的 rapidly 进步，人们对它们潜在的社会偏见和刻板印象学习与生成的担忧不断增加。关于 LVLM 的刻板印象的研究面临两大主要限制：忽视内容词重要性的指标和忽视颜色影响的数据集。为解决这些限制，本研究引入了基于刻板印象内容模型(Stereotype Content Model, SCM)的新评估指标，并提出了一项名为 BASIC 的基准测试，用于评估性别、种族和颜色刻板印象。使用 SCM 指标和 BASIC，我们对八种 LVLM 进行了研究，以发现刻板印象。结果，我们发现了三个发现：(1) 基于 SCM 的评估有效捕捉了刻板印象。(2) LVLMs 在输出中不仅表现出性别和种族刻板印象，还表现出颜色刻板印象。(3) 模型架构与参数大小之间的交互似乎影响刻板印象。我们已在 [审查中匿名化] 释放 BASIC。 

---
# In Context Learning with Vision Transformers: Case Study 

**Title (ZH)**: 基于视觉变换器的上下文学习：案例研究 

**Authors**: Antony Zhao, Alex Proshkin, Fergal Hennessy, Francesco Crivelli  

**Link**: [PDF](https://arxiv.org/pdf/2505.20872)  

**Abstract**: Large transformer models have been shown to be capable of performing in-context learning. By using examples in a prompt as well as a query, they are capable of performing tasks such as few-shot, one-shot, or zero-shot learning to output the corresponding answer to this query. One area of interest to us is that these transformer models have been shown to be capable of learning the general class of certain functions, such as linear functions and small 2-layer neural networks, on random data (Garg et al, 2023). We aim to extend this to the image space to analyze their capability to in-context learn more complex functions on the image space, such as convolutional neural networks and other methods. 

**Abstract (ZH)**: 大型变压器模型已被证明能够进行在上下文学习。通过在提示中使用示例和查询，它们能够执行少样本、单样本或零样本学习，输出相应的查询答案。我们感兴趣的一个领域是这些变压器模型能够从随机数据中学习某些函数类，如线性函数和小型2层神经网络（Garg等人，2023）。我们旨在将这一点扩展到图像空间，分析它们在图像空间中学习更复杂函数的能力，如卷积神经网络和其他方法。 

---
# Rendering-Aware Reinforcement Learning for Vector Graphics Generation 

**Title (ZH)**: 面向渲染的强化学习在矢量图形生成中的应用 

**Authors**: Juan A. Rodriguez, Haotian Zhang, Abhay Puri, Aarash Feizi, Rishav Pramanik, Pascal Wichmann, Arnab Mondal, Mohammad Reza Samsami, Rabiul Awal, Perouz Taslakian, Spandana Gella, Sai Rajeswar, David Vazquez, Christopher Pal, Marco Pedersoli  

**Link**: [PDF](https://arxiv.org/pdf/2505.20793)  

**Abstract**: Scalable Vector Graphics (SVG) offer a powerful format for representing visual designs as interpretable code. Recent advances in vision-language models (VLMs) have enabled high-quality SVG generation by framing the problem as a code generation task and leveraging large-scale pretraining. VLMs are particularly suitable for this task as they capture both global semantics and fine-grained visual patterns, while transferring knowledge across vision, natural language, and code domains. However, existing VLM approaches often struggle to produce faithful and efficient SVGs because they never observe the rendered images during training. Although differentiable rendering for autoregressive SVG code generation remains unavailable, rendered outputs can still be compared to original inputs, enabling evaluative feedback suitable for reinforcement learning (RL). We introduce RLRF(Reinforcement Learning from Rendering Feedback), an RL method that enhances SVG generation in autoregressive VLMs by leveraging feedback from rendered SVG outputs. Given an input image, the model generates SVG roll-outs that are rendered and compared to the original image to compute a reward. This visual fidelity feedback guides the model toward producing more accurate, efficient, and semantically coherent SVGs. RLRF significantly outperforms supervised fine-tuning, addressing common failure modes and enabling precise, high-quality SVG generation with strong structural understanding and generalization. 

**Abstract (ZH)**: 可扩展矢量图形 (SVG) 提供了一种强大的格式，用于以可解释的代码表示视觉设计。最近在视觉-语言模型（VLMs）方面的进展通过将问题框定为代码生成任务并利用大规模预训练，使高质量SVG生成成为可能。VLMs特别适合此任务，因为它们能够捕捉全局语义和细粒度的视觉模式，并在视觉、自然语言和代码领域之间转移知识。然而，现有的VLM方法在训练过程中通常无法生成忠实且高效的SVG，因为它们从未观察过渲染后的图像。尽管自回归SVG代码生成的可微分渲染尚未可用，但仍可以将渲染输出与原始输入进行对比，从而提供适用于强化学习（RL）的评估反馈。我们介绍了基于渲染反馈的RLRF（Reinforcement Learning from Rendering Feedback），一种利用渲染SVG输出反馈增强自回归VLM中SVG生成的RL方法。给定输入图像，该模型生成SVG滚出，渲染并与原始图像进行比较以计算奖励。这种视觉保真度反馈引导模型生成更准确、更高效且语义一致的SVG。RLRF 显著优于监督微调，解决了常见失败模式，并使生成具有强大结构理解和泛化能力的精确且高质量SVG成为可能。 

---
# Can we Debias Social Stereotypes in AI-Generated Images? Examining Text-to-Image Outputs and User Perceptions 

**Title (ZH)**: 我们能否在AI生成的图像中消除社会刻板印象？探究文本到图像输出及用户感知。 

**Authors**: Saharsh Barve, Andy Mao, Jiayue Melissa Shi, Prerna Juneja, Koustuv Saha  

**Link**: [PDF](https://arxiv.org/pdf/2505.20692)  

**Abstract**: Recent advances in generative AI have enabled visual content creation through text-to-image (T2I) generation. However, despite their creative potential, T2I models often replicate and amplify societal stereotypes -- particularly those related to gender, race, and culture -- raising important ethical concerns. This paper proposes a theory-driven bias detection rubric and a Social Stereotype Index (SSI) to systematically evaluate social biases in T2I outputs. We audited three major T2I model outputs -- DALL-E-3, Midjourney-6.1, and Stability AI Core -- using 100 queries across three categories -- geocultural, occupational, and adjectival. Our analysis reveals that initial outputs are prone to include stereotypical visual cues, including gendered professions, cultural markers, and western beauty norms. To address this, we adopted our rubric to conduct targeted prompt refinement using LLMs, which significantly reduced bias -- SSI dropped by 61% for geocultural, 69% for occupational, and 51% for adjectival queries. We complemented our quantitative analysis through a user study examining perceptions, awareness, and preferences around AI-generated biased imagery. Our findings reveal a key tension -- although prompt refinement can mitigate stereotypes, it can limit contextual alignment. Interestingly, users often perceived stereotypical images to be more aligned with their expectations. We discuss the need to balance ethical debiasing with contextual relevance and call for T2I systems that support global diversity and inclusivity while not compromising the reflection of real-world social complexity. 

**Abstract (ZH)**: recent 进展在生成式 AI 领域使得通过文本生成图像（T2I）创建视觉内容成为可能。然而，尽管 T2I 模型具有创造性潜力，它们往往会复制和放大与性别、种族和文化相关的社会刻板印象，引发了重要的伦理关切。本文提出了一种基于理论的偏见检测量表和社会刻板印象指数（SSI）来系统评估 T2I 输出中的社会偏见。我们使用 100 个查询，涵盖地理文化、职业和形容词三个类别，对三个主要的 T2I 模型输出——DALL-E-3、Midjourney-6.1 和 Stability AI Core 进行了审计。分析结果表明，初始输出倾向于包含刻板的视觉线索，包括性别化的职业、文化标志和西方美容规范。为了解决这个问题，我们采用我们的量表并利用大语言模型（LLMs）进行针对性的提示优化，这显著降低了偏见——地理文化的 SSI 下降了 61%，职业的 SSI 下降了 69%，形容词查询的 SSI 下降了 51%。我们通过一项用户研究补充了定量分析，该研究 examines 用户对 AI 生成的带有偏见的图像的感知、意识和偏好。我们的研究结果揭示了一个关键矛盾——尽管提示优化可以缓解刻板印象，但它可能会限制上下文一致性。有趣的是，用户往往认为刻板印象图像与他们的预期更一致。我们讨论了在伦理去偏见与上下文相关性之间取得平衡的必要性，并呼吁支持全球多样性和包容性的 T2I 系统，但同时不应牺牲对现实社会复杂性的反映。 

---
# HCQA-1.5 @ Ego4D EgoSchema Challenge 2025 

**Title (ZH)**: HCQA-1.5 @ Ego4D EgoSchema挑战赛2025 

**Authors**: Haoyu Zhang, Yisen Feng, Qiaohui Chu, Meng Liu, Weili Guan, Yaowei Wang, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2505.20644)  

**Abstract**: In this report, we present the method that achieves third place for Ego4D EgoSchema Challenge in CVPR 2025. To improve the reliability of answer prediction in egocentric video question answering, we propose an effective extension to the previously proposed HCQA framework. Our approach introduces a multi-source aggregation strategy to generate diverse predictions, followed by a confidence-based filtering mechanism that selects high-confidence answers directly. For low-confidence cases, we incorporate a fine-grained reasoning module that performs additional visual and contextual analysis to refine the predictions. Evaluated on the EgoSchema blind test set, our method achieves 77% accuracy on over 5,000 human-curated multiple-choice questions, outperforming last year's winning solution and the majority of participating teams. Our code will be added at this https URL. 

**Abstract (ZH)**: 在本报告中，我们展示了在CVPR 2025 Ego4D EgoSchema挑战中获得第三名的方法。为了提高自视点视频问答中答案预测的可靠性，我们提出了对先前提出的HCQA框架的有效扩展。我们的方法引入了多源聚合策略以生成多样化预测，并通过基于置信度的过滤机制直接选择高置信度的答案。对于低置信度情况，我们引入了细粒度推理模块，进行额外的视觉和情境分析以细化预测。在EgoSchema盲测试集上，我们的方法在超过5,000个人工策划的多选题中实现了77%的准确率，超过了去年的获胜解决方案和大多数参赛队伍。我们的代码将在此处 https:// 提供。 

---
# Plug-and-Play Co-Occurring Face Attention for Robust Audio-Visual Speaker Extraction 

**Title (ZH)**: 即插即用共现面孔注意力以实现鲁棒的音视频说话人提取 

**Authors**: Zexu Pan, Shengkui Zhao, Tingting Wang, Kun Zhou, Yukun Ma, Chong Zhang, Bin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.20635)  

**Abstract**: Audio-visual speaker extraction isolates a target speaker's speech from a mixture speech signal conditioned on a visual cue, typically using the target speaker's face recording. However, in real-world scenarios, other co-occurring faces are often present on-screen, providing valuable speaker activity cues in the scene. In this work, we introduce a plug-and-play inter-speaker attention module to process these flexible numbers of co-occurring faces, allowing for more accurate speaker extraction in complex multi-person environments. We integrate our module into two prominent models: the AV-DPRNN and the state-of-the-art AV-TFGridNet. Extensive experiments on diverse datasets, including the highly overlapped VoxCeleb2 and sparsely overlapped MISP, demonstrate that our approach consistently outperforms baselines. Furthermore, cross-dataset evaluations on LRS2 and LRS3 confirm the robustness and generalizability of our method. 

**Abstract (ZH)**: 基于视觉的说话人提取模块处理多个人环境中的灵活数量的共存面孔，以实现更准确的说话人隔离 

---
# InstGenIE: Generative Image Editing Made Efficient with Mask-aware Caching and Scheduling 

**Title (ZH)**: InstGenIE: 通过掩码感知缓存与调度实现高效的生成图像编辑 

**Authors**: Xiaoxiao Jiang, Suyi Li, Lingyun Yang, Tianyu Feng, Zhipeng Di, Weiyi Lu, Guoxuan Zhu, Xiu Lin, Kan Liu, Yinghao Yu, Tao Lan, Guodong Yang, Lin Qu, Liping Zhang, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20600)  

**Abstract**: Generative image editing using diffusion models has become a prevalent application in today's AI cloud services. In production environments, image editing typically involves a mask that specifies the regions of an image template to be edited. The use of masks provides direct control over the editing process and introduces sparsity in the model inference. In this paper, we present InstGenIE, a system that efficiently serves image editing requests. The key insight behind InstGenIE is that image editing only modifies the masked regions of image templates while preserving the original content in the unmasked areas. Driven by this insight, InstGenIE judiciously skips redundant computations associated with the unmasked areas by reusing cached intermediate activations from previous inferences. To mitigate the high cache loading overhead, InstGenIE employs a bubble-free pipeline scheme that overlaps computation with cache loading. Additionally, to reduce queuing latency in online serving while improving the GPU utilization, InstGenIE proposes a novel continuous batching strategy for diffusion model serving, allowing newly arrived requests to join the running batch in just one step of denoising computation, without waiting for the entire batch to complete. As heterogeneous masks induce imbalanced loads, InstGenIE also develops a load balancing strategy that takes into account the loads of both computation and cache loading. Collectively, InstGenIE outperforms state-of-the-art diffusion serving systems for image editing, achieving up to 3x higher throughput and reducing average request latency by up to 14.7x while ensuring image quality. 

**Abstract (ZH)**: 基于扩散模型的生成图像编辑系统InstGenIE 

---
# CCL-LGS: Contrastive Codebook Learning for 3D Language Gaussian Splatting 

**Title (ZH)**: CCL-LGS: 对比学习代码簿 для 3D 语言高斯点云生成 

**Authors**: Lei Tian, Xiaomin Li, Liqian Ma, Hefei Huang, Zirui Zheng, Hao Yin, Taiqing Li, Huchuan Lu, Xu Jia  

**Link**: [PDF](https://arxiv.org/pdf/2505.20469)  

**Abstract**: Recent advances in 3D reconstruction techniques and vision-language models have fueled significant progress in 3D semantic understanding, a capability critical to robotics, autonomous driving, and virtual/augmented reality. However, methods that rely on 2D priors are prone to a critical challenge: cross-view semantic inconsistencies induced by occlusion, image blur, and view-dependent variations. These inconsistencies, when propagated via projection supervision, deteriorate the quality of 3D Gaussian semantic fields and introduce artifacts in the rendered outputs. To mitigate this limitation, we propose CCL-LGS, a novel framework that enforces view-consistent semantic supervision by integrating multi-view semantic cues. Specifically, our approach first employs a zero-shot tracker to align a set of SAM-generated 2D masks and reliably identify their corresponding categories. Next, we utilize CLIP to extract robust semantic encodings across views. Finally, our Contrastive Codebook Learning (CCL) module distills discriminative semantic features by enforcing intra-class compactness and inter-class distinctiveness. In contrast to previous methods that directly apply CLIP to imperfect masks, our framework explicitly resolves semantic conflicts while preserving category discriminability. Extensive experiments demonstrate that CCL-LGS outperforms previous state-of-the-art methods. Our project page is available at this https URL. 

**Abstract (ZH)**: Recent advances in 3D重建技术和视觉语言模型促进了三维语义理解的重大进展，这一能力对机器人技术、自动驾驶和虚拟/增强现实至关重要。然而，依赖2D先验的方法易遭受关键挑战：因遮挡、图像模糊和视角依赖性变化引发的跨视角语义不一致性。这些不一致性通过投影监督传播，会降低3D高斯语义场的质量并在渲染输出中引入伪影。为了减轻这一限制，我们提出了一种名为CCL-LGS的新框架，通过集成多视角语义线索来确保视图一致的语义监督。具体而言，我们的方法首先采用零样本跟踪器对SAM生成的2D掩码进行对齐，并可靠地识别其对应的类别。接着，我们利用CLIP在不同视角中提取鲁棒的语义编码。最后，我们的对比码本学习（CCL）模块通过促进类别内紧凑性和类别间区分性来提炼具有辨别性的语义特征。与之前直接将CLIP应用于不完美的掩码的方法相比，我们的框架显式地解决了语义冲突并保留了类别的可辨别性。大量实验表明，CCL-LGS优于之前的方法。我们的项目页面可在此 https URL 查看。 

---
# MVTN: Learning Multi-View Transformations for 3D Understanding 

**Title (ZH)**: MVTN：学习多视图变换以进行三维理解 

**Authors**: Abdullah Hamdi, Faisal AlZahrani, Silvio Giancola, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2212.13462)  

**Abstract**: Multi-view projection techniques have shown themselves to be highly effective in achieving top-performing results in the recognition of 3D shapes. These methods involve learning how to combine information from multiple view-points. However, the camera view-points from which these views are obtained are often fixed for all shapes. To overcome the static nature of current multi-view techniques, we propose learning these view-points. Specifically, we introduce the Multi-View Transformation Network (MVTN), which uses differentiable rendering to determine optimal view-points for 3D shape recognition. As a result, MVTN can be trained end-to-end with any multi-view network for 3D shape classification. We integrate MVTN into a novel adaptive multi-view pipeline that is capable of rendering both 3D meshes and point clouds. Our approach demonstrates state-of-the-art performance in 3D classification and shape retrieval on several benchmarks (ModelNet40, ScanObjectNN, ShapeNet Core55). Further analysis indicates that our approach exhibits improved robustness to occlusion compared to other methods. We also investigate additional aspects of MVTN, such as 2D pretraining and its use for segmentation. To support further research in this area, we have released MVTorch, a PyTorch library for 3D understanding and generation using multi-view projections. 

**Abstract (ZH)**: 多视角投影技术在3D形状识别中表现出色，能够实现顶级性能。这些方法涉及学习如何从多个视角整合信息。然而，这些视角通常对于所有形状而言是固定的。为克服当前多视角技术的静态特性，我们提出学习这些视角。具体地，我们引入了多视角变换网络（MVTN），该网络利用可微渲染来确定3D形状识别的最优视角。因此，MVTN可以与任何多视角网络端到端地训练用于3D形状分类。我们将MVTN集成到一个新颖的自适应多视角管道中，该管道能够渲染3D网格和点云。我们的方法在多种基准测试（ModelNet40、ScanObjectNN、ShapeNet Core55）中的3D分类和形状检索中达到了最先进的性能。进一步的分析表明，与其它方法相比，我们的方法在遮挡鲁棒性方面有所提升。我们还研究了MVTN的其他方面，如2D预训练及其用于分割的应用。为了支持该领域的进一步研究，我们发布了MVTorch库，这是一个用于通过多视角投影进行3D理解和生成的PyTorch库。 

---
