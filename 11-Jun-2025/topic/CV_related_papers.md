# Diffusion Models for Safety Validation of Autonomous Driving Systems 

**Title (ZH)**: 自主驾驶系统安全性验证的扩散模型 

**Authors**: Juanran Wang, Marc R. Schlichting, Harrison Delecki, Mykel J. Kochenderfer  

**Link**: [PDF](https://arxiv.org/pdf/2506.08459)  

**Abstract**: Safety validation of autonomous driving systems is extremely challenging due to the high risks and costs of real-world testing as well as the rarity and diversity of potential failures. To address these challenges, we train a denoising diffusion model to generate potential failure cases of an autonomous vehicle given any initial traffic state. Experiments on a four-way intersection problem show that in a variety of scenarios, the diffusion model can generate realistic failure samples while capturing a wide variety of potential failures. Our model does not require any external training dataset, can perform training and inference with modest computing resources, and does not assume any prior knowledge of the system under test, with applicability to safety validation for traffic intersections. 

**Abstract (ZH)**: 自主驾驶系统安全验证由于现实世界测试的风险和成本高以及潜在故障的稀有性和多样性而极具挑战性。为应对这些挑战，我们训练一个去噪扩散模型，给定任意初始交通状态，生成该自主车辆的潜在故障案例。在四向交叉口问题上的实验表明，在多种场景下，扩散模型能够生成现实的故障样本，同时捕获广泛的潜在故障。我们的模型不需要任何外部训练数据集，可以使用有限的计算资源进行训练和推理，并且不需要测试系统的先验知识，适用于交通交叉口的安全验证。 

---
# Attention-based Learning for 3D Informative Path Planning 

**Title (ZH)**: 基于注意力的学习在3D信息性路径规划中的应用 

**Authors**: Rui Zhao, Xingjian Zhang, Yuhong Cao, Yizhuo Wang, Guillaume Sartoretti  

**Link**: [PDF](https://arxiv.org/pdf/2506.08434)  

**Abstract**: In this work, we propose an attention-based deep reinforcement learning approach to address the adaptive informative path planning (IPP) problem in 3D space, where an aerial robot equipped with a downward-facing sensor must dynamically adjust its 3D position to balance sensing footprint and accuracy, and finally obtain a high-quality belief of an underlying field of interest over a given domain (e.g., presence of specific plants, hazardous gas, geological structures, etc.). In adaptive IPP tasks, the agent is tasked with maximizing information collected under time/distance constraints, continuously adapting its path based on newly acquired sensor data. To this end, we leverage attention mechanisms for their strong ability to capture global spatial dependencies across large action spaces, allowing the agent to learn an implicit estimation of environmental transitions. Our model builds a contextual belief representation over the entire domain, guiding sequential movement decisions that optimize both short- and long-term search objectives. Comparative evaluations against state-of-the-art planners demonstrate that our approach significantly reduces environmental uncertainty within constrained budgets, thus allowing the agent to effectively balance exploration and exploitation. We further show our model generalizes well to environments of varying sizes, highlighting its potential for many real-world applications. 

**Abstract (ZH)**: 基于注意力机制的深度强化学习在三维空间自适应信息路径规划问题中的应用 

---
# SDTagNet: Leveraging Text-Annotated Navigation Maps for Online HD Map Construction 

**Title (ZH)**: SDTagNet: 利用文本标注的导航地图进行在线高精度地图构建 

**Authors**: Fabian Immel, Jan-Hendrik Pauls, Richard Fehler, Frank Bieder, Jonas Merkert, Christoph Stiller  

**Link**: [PDF](https://arxiv.org/pdf/2506.08997)  

**Abstract**: Autonomous vehicles rely on detailed and accurate environmental information to operate safely. High definition (HD) maps offer a promising solution, but their high maintenance cost poses a significant barrier to scalable deployment. This challenge is addressed by online HD map construction methods, which generate local HD maps from live sensor data. However, these methods are inherently limited by the short perception range of onboard sensors. To overcome this limitation and improve general performance, recent approaches have explored the use of standard definition (SD) maps as prior, which are significantly easier to maintain. We propose SDTagNet, the first online HD map construction method that fully utilizes the information of widely available SD maps, like OpenStreetMap, to enhance far range detection accuracy. Our approach introduces two key innovations. First, in contrast to previous work, we incorporate not only polyline SD map data with manually selected classes, but additional semantic information in the form of textual annotations. In this way, we enrich SD vector map tokens with NLP-derived features, eliminating the dependency on predefined specifications or exhaustive class taxonomies. Second, we introduce a point-level SD map encoder together with orthogonal element identifiers to uniformly integrate all types of map elements. Experiments on Argoverse 2 and nuScenes show that this boosts map perception performance by up to +5.9 mAP (+45%) w.r.t. map construction without priors and up to +3.2 mAP (+20%) w.r.t. previous approaches that already use SD map priors. Code is available at this https URL 

**Abstract (ZH)**: 自主驾驶车辆依赖于详细的accurate环境信息以确保安全运行。高精度(HD)地图提供了有希望的解决方案，但其高昂的维护成本成为广泛应用的瓶颈。通过在线HD地图构建方法，可以利用实时传感器数据生成局部HD地图，但这些方法受限于车载传感器较短的感知范围。为解决这一限制并提高整体性能，最近的研究探索了标准定义(SD)地图作为先验的可能性，SD地图易于维护得多。我们提出了SDTagNet，这是首个充分利用广泛可用的SD地图信息（例如OpenStreetMap）以增强远距离检测准确性的在线HD地图构建方法。我们的方法引入了两项关键创新。首先，与先前的工作不同，我们不仅利用带有人工选择类别的多段线SD地图数据，还附加了文本注释形式的语义信息，从而通过NLP提取特征丰富SD矢量地图标记，消除了对预定义规范或详尽类别分类的需求。其次，我们引入了一种点级别SD地图编码器，并结合正交元素标识符，以统一整合各类地图元素。实验结果表明，与不使用先验的HD地图构建方法相比，该方法的地图感知性能提高了最多5.9 mAP (+45%)；与已经使用SD地图先验的先前方法相比，提高了最多3.2 mAP (+20%)。代码可在此处访问。 

---
# Rethinking Range-View LiDAR Segmentation in Adverse Weather 

**Title (ZH)**: 重新思考不良天气条件下的范围视图LiDAR分割 

**Authors**: Longyu Yang, Ping Hu, Lu Zhang, Jun Liu, Yap-Peng Tan, Heng Tao Shen, Xiaofeng Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08979)  

**Abstract**: LiDAR segmentation has emerged as an important task to enrich multimedia experiences and analysis. Range-view-based methods have gained popularity due to their high computational efficiency and compatibility with real-time deployment. However, their generalized performance under adverse weather conditions remains underexplored, limiting their reliability in real-world environments. In this work, we identify and analyze the unique challenges that affect the generalization of range-view LiDAR segmentation in severe weather. To address these challenges, we propose a modular and lightweight framework that enhances robustness without altering the core architecture of existing models. Our method reformulates the initial stem block of standard range-view networks into two branches to process geometric attributes and reflectance intensity separately. Specifically, a Geometric Abnormality Suppression (GAS) module reduces the influence of weather-induced spatial noise, and a Reflectance Distortion Calibration (RDC) module corrects reflectance distortions through memory-guided adaptive instance normalization. The processed features are then fused and passed to the original segmentation pipeline. Extensive experiments on different benchmarks and baseline models demonstrate that our approach significantly improves generalization to adverse weather with minimal inference overhead, offering a practical and effective solution for real-world LiDAR segmentation. 

**Abstract (ZH)**: 基于范围视图的LiDAR分割在恶劣天气条件下的通用性分析与提升方法 

---
# Scaling Laws of Motion Forecasting and Planning -- A Technical Report 

**Title (ZH)**: 运动预测与规划的标度律——技术报告 

**Authors**: Mustafa Baniodeh, Kratarth Goel, Scott Ettinger, Carlos Fuertes, Ari Seff, Tim Shen, Cole Gulino, Chenjie Yang, Ghassen Jerfel, Dokook Choe, Rui Wang, Vinutha Kallem, Sergio Casas, Rami Al-Rfou, Benjamin Sapp, Dragomir Anguelov  

**Link**: [PDF](https://arxiv.org/pdf/2506.08228)  

**Abstract**: We study the empirical scaling laws of a family of encoder-decoder autoregressive transformer models on the task of joint motion forecasting and planning in the autonomous driving domain. Using a 500 thousand hours driving dataset, we demonstrate that, similar to language modeling, model performance improves as a power-law function of the total compute budget, and we observe a strong correlation between model training loss and model evaluation metrics. Most interestingly, closed-loop metrics also improve with scaling, which has important implications for the suitability of open-loop metrics for model development and hill climbing. We also study the optimal scaling of the number of transformer parameters and the training data size for a training compute-optimal model. We find that as the training compute budget grows, optimal scaling requires increasing the model size 1.5x as fast as the dataset size. We also study inference-time compute scaling, where we observe that sampling and clustering the output of smaller models makes them competitive with larger models, up to a crossover point beyond which a larger models becomes more inference-compute efficient. Overall, our experimental results demonstrate that optimizing the training and inference-time scaling properties of motion forecasting and planning models is a key lever for improving their performance to address a wide variety of driving scenarios. Finally, we briefly study the utility of training on general logged driving data of other agents to improve the performance of the ego-agent, an important research area to address the scarcity of robotics data for large capacity models training. 

**Abstract (ZH)**: 我们研究了一类编码器-解码器自回归变压器模型在自主驾驶领域联合运动预测与规划任务中的经验标度律。使用50万小时的驾驶数据集，我们表明，类似于语言建模，模型性能随着总计算预算的幂律函数提高，并观察到模型训练损失与模型评估指标之间存在强烈的相关性。最有趣的是，闭环指标也随着标度提高，这对开放环指标在模型开发和爬坡中的适用性具有重要意义。我们还研究了训练计算最优模型的变压器参数数量和训练数据规模的最佳标度。我们发现，随着训练计算预算的增长，最优标度需要将模型大小以比数据集大小快1.5倍的速度增加。我们还研究了推理时的计算标度，观察到对较小模型的输出进行采样和聚类使其与较大模型竞争，直到交叉点之后，较大模型在推理计算效率上更具优势。总体而言，我们的实验结果表明，优化运动预测与规划模型的训练和推理时的标度特性是提高其性能以应对各种驾驶场景的关键杠杆。最后，我们简要研究了利用其他代理的一般记录驾驶数据进行训练以改进自身代理性能的研究领域，这对于解决大规模模型训练中机器人数据稀缺问题具有重要意义。 

---
# Neural-Augmented Kelvinlet: Real-Time Soft Tissue Deformation with Multiple Graspers 

**Title (ZH)**: 神经增强Kelvinlet：多爪器实时软组织变形算法 

**Authors**: Ashkan Shahbazi, Kyvia Pereira, Jon S. Heiselman, Elaheh Akbari, Annie C. Benson, Sepehr Seifi, Xinyuan Liu, Garrison L. Johnston, Erwin Terpstra, Anne Draaisma, Jan-Jaap Severes, Jie Ying Wu, Nabil Simaan, Michael L.Miga, Soheil Kolouri  

**Link**: [PDF](https://arxiv.org/pdf/2506.08043)  

**Abstract**: Fast and accurate simulation of soft tissue deformation is a critical factor for surgical robotics and medical training. In this paper, we introduce a novel physics-informed neural simulator that approximates soft tissue deformations in a realistic and real-time manner. Our framework integrates Kelvinlet-based priors into neural simulators, making it the first approach to leverage Kelvinlets for residual learning and regularization in data-driven soft tissue modeling. By incorporating large-scale Finite Element Method (FEM) simulations of both linear and nonlinear soft tissue responses, our method improves neural network predictions across diverse architectures, enhancing accuracy and physical consistency while maintaining low latency for real-time performance. We demonstrate the effectiveness of our approach by performing accurate surgical maneuvers that simulate the use of standard laparoscopic tissue grasping tools with high fidelity. These results establish Kelvinlet-augmented learning as a powerful and efficient strategy for real-time, physics-aware soft tissue simulation in surgical applications. 

**Abstract (ZH)**: 快速且准确的软组织变形模拟是手术机器人和医疗培训的关键因素。本文介绍了一种新颖的物理信息神经模拟器，能够在现实且实时的条件下近似软组织变形。我们的框架将Kelvinlet基础先验整合到神经模拟器中，使其成为首次利用Kelvinlets进行残差学习和数据驱动软组织建模正则化的研究方法。通过整合大规模的有限元方法(FEM)模拟，我们的方法在多种架构的神经网络预测中得到了改进，增强了准确性和物理一致性，同时保持了低延迟以实现实时性能。我们通过模拟标准腹腔镜组织抓取工具使用高保真的手术操作展示了我们方法的有效性。这些结果建立了Kelvinlet增强学习作为一种强大且高效的策略，用于手术应用中的实时、物理感知软组织模拟。 

---
# Evaluating Generative Vehicle Trajectory Models for Traffic Intersection Dynamics 

**Title (ZH)**: 评估生成式车辆轨迹模型在交通交叉口动态中的应用 

**Authors**: Yash Ranjan, Rahul Sengupta, Anand Rangarajan, Sanjay Ranka  

**Link**: [PDF](https://arxiv.org/pdf/2506.08963)  

**Abstract**: Traffic Intersections are vital to urban road networks as they regulate the movement of people and goods. However, they are regions of conflicting trajectories and are prone to accidents. Deep Generative models of traffic dynamics at signalized intersections can greatly help traffic authorities better understand the efficiency and safety aspects. At present, models are evaluated on computational metrics that primarily look at trajectory reconstruction errors. They are not evaluated online in a `live' microsimulation scenario. Further, these metrics do not adequately consider traffic engineering-specific concerns such as red-light violations, unallowed stoppage, etc. In this work, we provide a comprehensive analytics tool to train, run, and evaluate models with metrics that give better insights into model performance from a traffic engineering point of view. We train a state-of-the-art multi-vehicle trajectory forecasting model on a large dataset collected by running a calibrated scenario of a real-world urban intersection. We then evaluate the performance of the prediction models, online in a microsimulator, under unseen traffic conditions. We show that despite using ideally-behaved trajectories as input, and achieving low trajectory reconstruction errors, the generated trajectories show behaviors that break traffic rules. We introduce new metrics to evaluate such undesired behaviors and present our results. 

**Abstract (ZH)**: 交通交叉口是城市道路网络中的关键部分，它们调节着人流和物流的流动。然而，它们是轨迹冲突的区域，容易发生事故。能够生成信号控制交叉口交通动态的深度生成模型可以大大帮助交通管理部门更好地理解和提高效率及安全性。目前，这些模型主要根据计算指标进行评估，这些指标主要关注轨迹重构误差，并未在线评估在实时微观模拟场景中的表现。此外，这些指标未能充分考虑到交通工程相关的特定关注点，如闯红灯、非法停车等。在本研究中，我们提供了一个综合分析工具，用于训练、运行和评估模型，这些模型采用从真实城市交叉口校准场景中收集的大规模数据集进行训练，并从交通工程的角度更好地评估模型性能。我们训练了一个最先进的多车辆轨迹预测模型，并在线评估其在未见过的交通条件下的性能。尽管输入了理想行为的轨迹，并且实现了低轨迹重构误差，生成的轨迹仍然表现出违反交通规则的行为。我们引入了新的评估指标来评价这种不良行为，并展示了我们的研究成果。 

---
# FloorplanMAE:A self-supervised framework for complete floorplan generation from partial inputs 

**Title (ZH)**: FloorplanMAE：一种基于部分输入的完全平面图自监督生成框架 

**Authors**: Jun Yin, Jing Zhong, Pengyu Zeng, Peilin Li, Miao Zhang, Ran Luo, Shuai Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08363)  

**Abstract**: In the architectural design process, floorplan design is often a dynamic and iterative process. Architects progressively draw various parts of the floorplan according to their ideas and requirements, continuously adjusting and refining throughout the design process. Therefore, the ability to predict a complete floorplan from a partial one holds significant value in the design process. Such prediction can help architects quickly generate preliminary designs, improve design efficiency, and reduce the workload associated with repeated modifications. To address this need, we propose FloorplanMAE, a self-supervised learning framework for restoring incomplete floor plans into complete ones. First, we developed a floor plan reconstruction dataset, FloorplanNet, specifically trained on architectural floor plans. Secondly, we propose a floor plan reconstruction method based on Masked Autoencoders (MAE), which reconstructs missing parts by masking sections of the floor plan and training a lightweight Vision Transformer (ViT). We evaluated the reconstruction accuracy of FloorplanMAE and compared it with state-of-the-art benchmarks. Additionally, we validated the model using real sketches from the early stages of architectural design. Experimental results show that the FloorplanMAE model can generate high-quality complete floor plans from incomplete partial plans. This framework provides a scalable solution for floor plan generation, with broad application prospects. 

**Abstract (ZH)**: 在建筑设计过程中，平面图设计通常是动态和迭代的过程。设计者根据自己的理念和需求逐步绘制平面图的不同部分，并在整个设计过程中不断调整和优化。因此，从不完整的平面图预测完整的平面图的能力在设计过程中具有重要意义。这种预测可以帮助设计者快速生成初步设计，提高设计效率，并减少重复修改的工作量。为应对这一需求，我们提出FloorplanMAE，这是一种自监督学习框架，用于将不完整的平面图恢复为完整的平面图。首先，我们开发了一个专门针对建筑平面图训练的平面图重建数据集FloorplanNet。其次，我们提出了一种基于Masked Autoencoders (MAE)的平面图重建方法，通过掩蔽平面图的部分区域并训练轻量级的Vision Transformer (ViT)来重建缺失的部分。我们评估了FloorplanMAE的重建精度，并与最新的基准进行了对比。此外，我们还使用了建筑设计初期的真实草图验证了该模型。实验结果表明，FloorplanMAE模型可以从不完整的部分平面图生成高质量的完整平面图。该框架为平面图生成提供了可扩展的解决方案，具有广泛的应用前景。 

---
# Autoregressive Semantic Visual Reconstruction Helps VLMs Understand Better 

**Title (ZH)**: 自回归语义视觉重建有助于提升大模型的理解能力 

**Authors**: Dianyi Wang, Wei Song, Yikun Wang, Siyuan Wang, Kaicheng Yu, Zhongyu Wei, Jiaqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09040)  

**Abstract**: Typical large vision-language models (LVLMs) apply autoregressive supervision solely to textual sequences, without fully incorporating the visual modality into the learning process. This results in three key limitations: (1) an inability to utilize images without accompanying captions, (2) the risk that captions omit critical visual details, and (3) the challenge that certain vision-centric content cannot be adequately conveyed through text. As a result, current LVLMs often prioritize vision-to-language alignment while potentially overlooking fine-grained visual information. While some prior works have explored autoregressive image generation, effectively leveraging autoregressive visual supervision to enhance image understanding remains an open challenge. In this paper, we introduce Autoregressive Semantic Visual Reconstruction (ASVR), which enables joint learning of visual and textual modalities within a unified autoregressive framework. We show that autoregressively reconstructing the raw visual appearance of images does not enhance and may even impair multimodal understanding. In contrast, autoregressively reconstructing the semantic representation of images consistently improves comprehension. Notably, we find that even when models are given continuous image features as input, they can effectively reconstruct discrete semantic tokens, resulting in stable and consistent improvements across a wide range of multimodal understanding benchmarks. Our approach delivers significant performance gains across varying data scales (556k-2M) and types of LLM bacbones. Specifically, ASVR improves LLaVA-1.5 by 5% in average scores across 14 multimodal benchmarks. The code is available at this https URL. 

**Abstract (ZH)**: 典型的大型视觉-语言模型（LVLMs）仅对文本序列应用自回归监督，而未能充分将视觉模态整合到学习过程中。这导致了三个关键局限性：（1）无法利用未配对字幕的图像，（2）字幕可能遗漏关键的视觉细节，以及（3）某些视觉中心的内容难以通过文本充分表达。因此，当前的LVLMs往往在图像到语言对齐方面占优，但可能忽视了精细的视觉信息。虽然一些先前的工作探索了自回归图像生成，但如何有效利用自回归的视觉监督来增强图像理解仍然是一个开放的挑战。在本文中，我们引入了自回归语义视觉重建（ASVR），使其能够在统一的自回归框架中联合学习视觉和文本模态。我们表明，自回归重建图像的原始视觉外观并未提升多模态理解，甚至可能损害多模态理解。相反，自回归重建图像的语义表示始终能提高理解能力。值得注意的是，即使给模型提供连续的图像特征作为输入，它们也能有效地重建离散的语义令牌，从而在多种多模态理解基准测试中实现稳定且一致的改进。我们的方法在不同数据规模（556k-2M）和不同类型的大规模语言模型（LLM）底座上实现了显著的性能提升。具体而言，ASVR使LLaVA-1.5在14个不同多模态基准测试中的平均得分提高了5%。代码可在以下链接获得。 

---
# Diffuse and Disperse: Image Generation with Representation Regularization 

**Title (ZH)**: 扩散与分散：基于表示正则化的图像生成 

**Authors**: Runqian Wang, Kaiming He  

**Link**: [PDF](https://arxiv.org/pdf/2506.09027)  

**Abstract**: The development of diffusion-based generative models over the past decade has largely proceeded independently of progress in representation learning. These diffusion models typically rely on regression-based objectives and generally lack explicit regularization. In this work, we propose \textit{Dispersive Loss}, a simple plug-and-play regularizer that effectively improves diffusion-based generative models. Our loss function encourages internal representations to disperse in the hidden space, analogous to contrastive self-supervised learning, with the key distinction that it requires no positive sample pairs and therefore does not interfere with the sampling process used for regression. Compared to the recent method of representation alignment (REPA), our approach is self-contained and minimalist, requiring no pre-training, no additional parameters, and no external data. We evaluate Dispersive Loss on the ImageNet dataset across a range of models and report consistent improvements over widely used and strong baselines. We hope our work will help bridge the gap between generative modeling and representation learning. 

**Abstract (ZH)**: 过去十年基于扩散的生成模型的发展很大程度上与表示学习的进步独立进行。这些扩散模型通常依赖于基于回归的目标函数，并且通常缺乏明确的正则化。在本文中，我们提出了一种简单直观的正则化方法——分散损失（Dispersive Loss），它可以有效提升基于扩散的生成模型。我们的损失函数鼓励内部表示在隐空间中分散，类似于对比自监督学习，但关键区别在于它不需要正样本对，因此不会干扰用于回归的采样过程。与最近的表示对齐方法（REPA）相比，我们的方法是自包含且简约的，无需预训练、额外参数和外部数据。我们在ImageNet数据集上对多种模型应用分散损失，并报告了相对于广泛使用的强基线模型的一致改进。我们希望我们的工作能帮助弥合生成建模与表示学习之间的差距。 

---
# Segment Concealed Objects with Incomplete Supervision 

**Title (ZH)**: 用不完备监督揭示隐藏对象 

**Authors**: Chunming He, Kai Li, Yachao Zhang, Ziyun Yang, Youwei Pang, Longxiang Tang, Chengyu Fang, Yulun Zhang, Linghe Kong, Xiu Li, Sina Farsiu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08955)  

**Abstract**: Incompletely-Supervised Concealed Object Segmentation (ISCOS) involves segmenting objects that seamlessly blend into their surrounding environments, utilizing incompletely annotated data, such as weak and semi-annotations, for model training. This task remains highly challenging due to (1) the limited supervision provided by the incompletely annotated training data, and (2) the difficulty of distinguishing concealed objects from the background, which arises from the intrinsic similarities in concealed scenarios. In this paper, we introduce the first unified method for ISCOS to address these challenges. To tackle the issue of incomplete supervision, we propose a unified mean-teacher framework, SEE, that leverages the vision foundation model, ``\emph{Segment Anything Model (SAM)}'', to generate pseudo-labels using coarse masks produced by the teacher model as prompts. To mitigate the effect of low-quality segmentation masks, we introduce a series of strategies for pseudo-label generation, storage, and supervision. These strategies aim to produce informative pseudo-labels, store the best pseudo-labels generated, and select the most reliable components to guide the student model, thereby ensuring robust network training. Additionally, to tackle the issue of intrinsic similarity, we design a hybrid-granularity feature grouping module that groups features at different granularities and aggregates these results. By clustering similar features, this module promotes segmentation coherence, facilitating more complete segmentation for both single-object and multiple-object images. We validate the effectiveness of our approach across multiple ISCOS tasks, and experimental results demonstrate that our method achieves state-of-the-art performance. Furthermore, SEE can serve as a plug-and-play solution, enhancing the performance of existing models. 

**Abstract (ZH)**: 不完全监督隐藏对象分割（ISCOS）涉及利用不完全注释数据（如弱注释和半注释）对融合其环境中的对象进行分割，该任务由于（1）不完全注释训练数据提供的有限监督，以及（2）难以区分隐藏对象与背景（这是由于隐藏场景中固有的相似性）而具有高度挑战性。本文介绍了首个统一方法以应对这些挑战。为解决不完全监督的问题，我们提出了一种结合教师模型的统一教师框架SEE，利用“Segment Anything Model (SAM)”视觉基础模型生成伪标签，使用教师模型生成的粗略掩码作为提示。为减轻低质量分割掩码的影响，我们引入了一系列伪标签生成、存储和监督策略，旨在生成具有信息性的伪标签、存储最佳伪标签，并选择最可靠的组件来引导学生模型，从而确保网络训练的鲁棒性。此外，为应对固有相似性问题，我们设计了一个混合粒度特征分组模块，该模块在不同粒度下分组特征并聚合这些结果。通过聚类相似特征，该模块促进了分割的一致性，有助于对单对象和多对象图像进行更完整的分割。我们在多个ISCOS任务上验证了该方法的有效性，并且实验结果表明，我们的方法达到了最新性能。此外，SEE可以作为一种即插即用的解决方案，提升现有模型的性能。 

---
# Socratic-MCTS: Test-Time Visual Reasoning by Asking the Right Questions 

**Title (ZH)**: 苏格拉底-MCTS: 在线视觉推理通过提出正确的问题 

**Authors**: David Acuna, Ximing Lu, Jaehun Jung, Hyunwoo Kim, Amlan Kar, Sanja Fidler, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2506.08927)  

**Abstract**: Recent research in vision-language models (VLMs) has centered around the possibility of equipping them with implicit long-form chain-of-thought reasoning -- akin to the success observed in language models -- via distillation and reinforcement learning. But what about the non-reasoning models already trained and deployed across the internet? Should we simply abandon them, or is there hope for a search mechanism that can elicit hidden knowledge and induce long reasoning traces -- without any additional training or supervision? In this paper, we explore this possibility using a Monte Carlo Tree Search (MCTS)-inspired algorithm, which injects subquestion-subanswer pairs into the model's output stream. We show that framing reasoning as a search process -- where subquestions act as latent decisions within a broader inference trajectory -- helps the model "connect the dots" between fragmented knowledge and produce extended reasoning traces in non-reasoning models. We evaluate our method across three benchmarks and observe consistent improvements. Notably, our approach yields a 2% overall improvement on MMMU-PRO, including a significant 9% gain in Liberal Arts. 

**Abstract (ZH)**: 近期视觉-语言模型的研究集中于通过蒸馏和强化学习赋予它们潜在的长链推理能力，类似于语言模型的成功。但互联网上已训练和部署的非推理模型呢？我们是否应该完全放弃它们，还是有可能找到一种检索机制，能够在无需额外训练或监督的情况下唤起隐藏知识并诱导长推理痕迹？本文采用受蒙特卡洛树搜索(MCTS)启发的算法，在模型的输出流中注入子问题-子答对，将推理视为一个搜索过程，其中子问题作为广泛推理轨迹中的潜在决策，帮助模型连接碎片化的知识并生成非推理模型中的扩展推理痕迹。我们通过三个基准测试评估该方法，并观察到一致的改进。值得注意的是，我们的方法在MMMU-PRO上总体提高了2%，在人文学科方面取得了显著的9%的提升。 

---
# Inherently Faithful Attention Maps for Vision Transformers 

**Title (ZH)**: 内在忠实注意力图谱：用于视觉Transformer 

**Authors**: Ananthu Aniraj, Cassio F. Dantas, Dino Ienco, Diego Marcos  

**Link**: [PDF](https://arxiv.org/pdf/2506.08915)  

**Abstract**: We introduce an attention-based method that uses learned binary attention masks to ensure that only attended image regions influence the prediction. Context can strongly affect object perception, sometimes leading to biased representations, particularly when objects appear in out-of-distribution backgrounds. At the same time, many image-level object-centric tasks require identifying relevant regions, often requiring context. To address this conundrum, we propose a two-stage framework: stage 1 processes the full image to discover object parts and identify task-relevant regions, while stage 2 leverages input attention masking to restrict its receptive field to these regions, enabling a focused analysis while filtering out potentially spurious information. Both stages are trained jointly, allowing stage 2 to refine stage 1. Extensive experiments across diverse benchmarks demonstrate that our approach significantly improves robustness against spurious correlations and out-of-distribution backgrounds. 

**Abstract (ZH)**: 基于注意力的方法：使用学习的二元注意力掩模确保仅关注的图像区域影响预测 

---
# Product of Experts for Visual Generation 

**Title (ZH)**: 专家系统相乘方法在视觉生成中的应用 

**Authors**: Yunzhi Zhang, Carson Murtuza-Lanier, Zizhang Li, Yilun Du, Jiajun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08894)  

**Abstract**: Modern neural models capture rich priors and have complementary knowledge over shared data domains, e.g., images and videos. Integrating diverse knowledge from multiple sources -- including visual generative models, visual language models, and sources with human-crafted knowledge such as graphics engines and physics simulators -- remains under-explored. We propose a Product of Experts (PoE) framework that performs inference-time knowledge composition from heterogeneous models. This training-free approach samples from the product distribution across experts via Annealed Importance Sampling (AIS). Our framework shows practical benefits in image and video synthesis tasks, yielding better controllability than monolithic methods and additionally providing flexible user interfaces for specifying visual generation goals. 

**Abstract (ZH)**: 现代神经模型捕获丰富的先验知识并在共享数据域（如图像和视频）上互补。从多种来源综合多样知识——包括视觉生成模型、视觉语言模型以及包含人工构建知识（如图形引擎和物理模拟器）的来源——仍然未被充分探索。我们提出了一种专家乘积（PoE）框架，在推断时从异构模型中进行知识综合。这种无需训练的方法通过退火重要性采样（AIS）从专家产品的分布中采样。我们的框架在图像和视频合成任务中展示了实际优势，提供了比单体方法更好的可控性，并且还提供了灵活的用户界面来指定视觉生成目标。 

---
# Optimizing Learned Image Compression on Scalar and Entropy-Constraint Quantization 

**Title (ZH)**: 基于标量和熵约束量化的学习图像压缩优化 

**Authors**: Florian Borzechowski, Michael Schäfer, Heiko Schwarz, Jonathan Pfaff, Detlev Marpe, Thomas Wiegand  

**Link**: [PDF](https://arxiv.org/pdf/2506.08662)  

**Abstract**: The continuous improvements on image compression with variational autoencoders have lead to learned codecs competitive with conventional approaches in terms of rate-distortion efficiency. Nonetheless, taking the quantization into account during the training process remains a problem, since it produces zero derivatives almost everywhere and needs to be replaced with a differentiable approximation which allows end-to-end optimization. Though there are different methods for approximating the quantization, none of them model the quantization noise correctly and thus, result in suboptimal networks. Hence, we propose an additional finetuning training step: After conventional end-to-end training, parts of the network are retrained on quantized latents obtained at the inference stage. For entropy-constraint quantizers like Trellis-Coded Quantization, the impact of the quantizer is particularly difficult to approximate by rounding or adding noise as the quantized latents are interdependently chosen through a trellis search based on both the entropy model and a distortion measure. We show that retraining on correctly quantized data consistently yields additional coding gain for both uniform scalar and especially for entropy-constraint quantization, without increasing inference complexity. For the Kodak test set, we obtain average savings between 1% and 2%, and for the TecNick test set up to 2.2% in terms of Bjøntegaard-Delta bitrate. 

**Abstract (ZH)**: 基于变分自编码器的图像压缩连续改进已在率-失真效率方面使learned编解码器与传统方法竞争。然而，训练过程中考虑量化问题依然存在，因为量化过程几乎处处产生零导数，需要使用可微近似替代，从而实现端到端优化。尽管有多种方法近似量化过程，但 none 能够正确建模量化噪声，因此导致性能不佳的网络。因此，我们提出了一种额外的微调训练步骤：在传统端到端训练之后，重新训练网络的部分模块，使用推理阶段获得的量化潜在变量。对于基于熵约束的量化器（如梯形编码量化），量化器的影响通过舍入或添加噪声难以近似，因为量化潜在变量是基于熵模型和失真度量通过递归搜索相互选择的。我们表明，使用正确量化数据重新训练可以一致地为均匀标量量化和特别是熵约束量化提供额外的编码增益，且不增加推理复杂度。在 Kodak 测试集中，我们获得了平均 1% 到 2% 的比特率节省，而在 TecNick 测试集中，节省高达 2.2% 的 Bjøntegaard-Delta 比特率。 

---
# Time Series Representations for Classification Lie Hidden in Pretrained Vision Transformers 

**Title (ZH)**: 预训练视觉变换器中隐含的时间序列表示用于分类 

**Authors**: Simon Roschmann, Quentin Bouniot, Vasilii Feofanov, Ievgen Redko, Zeynep Akata  

**Link**: [PDF](https://arxiv.org/pdf/2506.08641)  

**Abstract**: Time series classification is a fundamental task in healthcare and industry, yet the development of time series foundation models (TSFMs) remains limited by the scarcity of publicly available time series datasets. In this work, we propose Time Vision Transformer (TiViT), a framework that converts time series into images to leverage the representational power of frozen Vision Transformers (ViTs) pretrained on large-scale image datasets. First, we theoretically motivate our approach by analyzing the 2D patching of ViTs for time series, showing that it can increase the number of label-relevant tokens and reduce the sample complexity. Second, we empirically demonstrate that TiViT achieves state-of-the-art performance on standard time series classification benchmarks by utilizing the hidden representations of large OpenCLIP models. We explore the structure of TiViT representations and find that intermediate layers with high intrinsic dimension are the most effective for time series classification. Finally, we assess the alignment between TiViT and TSFM representation spaces and identify a strong complementarity, with further performance gains achieved by combining their features. Our findings reveal yet another direction for reusing vision representations in a non-visual domain. 

**Abstract (ZH)**: 时间序列分类是医疗保健和工业中的一个基础任务，但由于公共时间序列数据集的稀缺性，时间序列基础模型（TSFMs）的发展仍然受到限制。在本文中，我们提出了一种名为Time Vision Transformer（TiViT）的框架，该框架将时间序列转换为图像，以利用在大规模图像数据集上预训练的冻结视觉变换器（ViTs）的表征能力。首先，我们从理论上通过分析ViTs的时间序列2D分块方法来阐释我们的方法，表明它可以增加与标签相关的标记的数量并降低样本复杂度。其次，我们通过利用大型OpenCLIP模型的隐藏表示，在标准的时间序列分类基准上展现了TiViT达到最先进的性能。我们探讨了TiViT表示结构，发现具有较高固有维度的中间层对时间序列分类最为有效。最后，我们在TiViT和TSFM表示空间之间进行了对齐评估，并发现两者之间存在强烈的互补性，结合它们的特征可以进一步提升性能。我们的研究结果揭示了在非视觉领域重新利用视觉表示的另一个方向。 

---
# ECMNet:Lightweight Semantic Segmentation with Efficient CNN-Mamba Network 

**Title (ZH)**: ECMNet：高效CNN-Mamba网络的轻量级语义分割 

**Authors**: Feixiang Du, Shengkun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08629)  

**Abstract**: In the past decade, Convolutional Neural Networks (CNNs) and Transformers have achieved wide applicaiton in semantic segmentation tasks. Although CNNs with Transformer models greatly improve performance, the global context modeling remains inadequate. Recently, Mamba achieved great potential in vision tasks, showing its advantages in modeling long-range dependency. In this paper, we propose a lightweight Efficient CNN-Mamba Network for semantic segmentation, dubbed as ECMNet. ECMNet combines CNN with Mamba skillfully in a capsule-based framework to address their complementary weaknesses. Specifically, We design a Enhanced Dual-Attention Block (EDAB) for lightweight bottleneck. In order to improve the representations ability of feature, We devise a Multi-Scale Attention Unit (MSAU) to integrate multi-scale feature aggregation, spatial aggregation and channel aggregation. Moreover, a Mamba enhanced Feature Fusion Module (FFM) merges diverse level feature, significantly enhancing segmented accuracy. Extensive experiments on two representative datasets demonstrate that the proposed model excels in accuracy and efficiency balance, achieving 70.6% mIoU on Cityscapes and 73.6% mIoU on CamVid test datasets, with 0.87M parameters and 8.27G FLOPs on a single RTX 3090 GPU platform. 

**Abstract (ZH)**: 基于Mamba的高效CNN网络及其在语义分割中的应用：ECMNet 

---
# TrajFlow: Multi-modal Motion Prediction via Flow Matching 

**Title (ZH)**: TrajFlow: 通过流匹配的多模态运动预测 

**Authors**: Qi Yan, Brian Zhang, Yutong Zhang, Daniel Yang, Joshua White, Di Chen, Jiachao Liu, Langechuan Liu, Binnan Zhuang, Shaoshuai Shi, Renjie Liao  

**Link**: [PDF](https://arxiv.org/pdf/2506.08541)  

**Abstract**: Efficient and accurate motion prediction is crucial for ensuring safety and informed decision-making in autonomous driving, particularly under dynamic real-world conditions that necessitate multi-modal forecasts. We introduce TrajFlow, a novel flow matching-based motion prediction framework that addresses the scalability and efficiency challenges of existing generative trajectory prediction methods. Unlike conventional generative approaches that employ i.i.d. sampling and require multiple inference passes to capture diverse outcomes, TrajFlow predicts multiple plausible future trajectories in a single pass, significantly reducing computational overhead while maintaining coherence across predictions. Moreover, we propose a ranking loss based on the Plackett-Luce distribution to improve uncertainty estimation of predicted trajectories. Additionally, we design a self-conditioning training technique that reuses the model's own predictions to construct noisy inputs during a second forward pass, thereby improving generalization and accelerating inference. Extensive experiments on the large-scale Waymo Open Motion Dataset (WOMD) demonstrate that TrajFlow achieves state-of-the-art performance across various key metrics, underscoring its effectiveness for safety-critical autonomous driving applications. The code and other details are available on the project website this https URL. 

**Abstract (ZH)**: 基于流匹配的高效准确运动预测框架TrajFlow在自动驾驶中的应用 

---
# DCD: A Semantic Segmentation Model for Fetal Ultrasound Four-Chamber View 

**Title (ZH)**: DCD：胎儿超声四腔观的语义分割模型 

**Authors**: Donglian Li, Hui Guo, Minglang Chen, Huizhen Chen, Jialing Chen, Bocheng Liang, Pengchen Liang, Ying Tan  

**Link**: [PDF](https://arxiv.org/pdf/2506.08534)  

**Abstract**: Accurate segmentation of anatomical structures in the apical four-chamber (A4C) view of fetal echocardiography is essential for early diagnosis and prenatal evaluation of congenital heart disease (CHD). However, precise segmentation remains challenging due to ultrasound artifacts, speckle noise, anatomical variability, and boundary ambiguity across different gestational stages. To reduce the workload of sonographers and enhance segmentation accuracy, we propose DCD, an advanced deep learning-based model for automatic segmentation of key anatomical structures in the fetal A4C view. Our model incorporates a Dense Atrous Spatial Pyramid Pooling (Dense ASPP) module, enabling superior multi-scale feature extraction, and a Convolutional Block Attention Module (CBAM) to enhance adaptive feature representation. By effectively capturing both local and global contextual information, DCD achieves precise and robust segmentation, contributing to improved prenatal cardiac assessment. 

**Abstract (ZH)**: 胎儿四腔观(A4C)心超中解剖结构的准确分割对于先天性心脏疾病(CHD)的早期诊断和产前评估至关重要。然而，由于超声伪像、speckle噪声、解剖结构的变异性和不同妊娠阶段边界模糊性，精确分割仍具有挑战性。为减轻超声操作者的负担并提高分割准确性，我们提出了一种基于深度学习的DCD先进模型，用于自动分割胎儿四腔观的关键解剖结构。该模型结合了 Dense Atrous Spatial Pyramid Pooling (Dense ASPP) 模块，实现了优异的多尺度特征提取，并使用 Convolutional Block Attention Module (CBAM) 来增强自适应特征表示。通过有效捕捉局部和全局上下文信息，DCD 实现了精确和鲁棒的分割，有助于提高产前心脏评估。 

---
# MLVTG: Mamba-Based Feature Alignment and LLM-Driven Purification for Multi-Modal Video Temporal Grounding 

**Title (ZH)**: MLVTG: 基于Mamba的特征对齐与基于LLM的多模态视频 temporal 基准净化 

**Authors**: Zhiyi Zhu, Xiaoyu Wu, Zihao Liu, Linlin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08512)  

**Abstract**: Video Temporal Grounding (VTG), which aims to localize video clips corresponding to natural language queries, is a fundamental yet challenging task in video understanding. Existing Transformer-based methods often suffer from redundant attention and suboptimal multi-modal alignment. To address these limitations, we propose MLVTG, a novel framework that integrates two key modules: MambaAligner and LLMRefiner. MambaAligner uses stacked Vision Mamba blocks as a backbone instead of Transformers to model temporal dependencies and extract robust video representations for multi-modal alignment. LLMRefiner leverages the specific frozen layer of a pre-trained Large Language Model (LLM) to implicitly transfer semantic priors, enhancing multi-modal alignment without fine-tuning. This dual alignment strategy, temporal modeling via structured state-space dynamics and semantic purification via textual priors, enables more precise localization. Extensive experiments on QVHighlights, Charades-STA, and TVSum demonstrate that MLVTG achieves state-of-the-art performance and significantly outperforms existing baselines. 

**Abstract (ZH)**: 视频时间定位（VTG），旨在定位与自然语言查询对应的视频片段，是视频理解中一个基础但具有挑战性的任务。现有的基于Transformer的方法常常受到冗余注意力和次优化多模态对齐的问题。为了解决这些局限，我们提出了MLVTG，一种新颖的框架，整合了两个关键模块：MambaAligner和LLMRefiner。MambaAligner使用堆叠的Vision Mamba块作为骨干，而不是Transformer，以建模时间依赖关系并提取用于多模态对齐的稳健视频表示。LLMRefiner利用预训练大型语言模型（LLM）中特定冻结层来隐式传递语义先验，增强多模态对齐而无需微调。这种双重对齐策略，通过结构化状态空间动力学建模时间关系并通过文本先验净化语义，能够实现更精确的定位。在QVHighlights、Charades-STA和TVSum上的广泛实验表明，MLVTG达到最佳性能并显著优于现有基线。 

---
# Spatiotemporal deep learning models for detection of rapid intensification in cyclones 

**Title (ZH)**: 基于时空深度学习模型的cyclone快速增强检测方法 

**Authors**: Vamshika Sutar, Amandeep Singh, Rohitash Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2506.08397)  

**Abstract**: Cyclone rapid intensification is the rapid increase in cyclone wind intensity, exceeding a threshold of 30 knots, within 24 hours. Rapid intensification is considered an extreme event during a cyclone, and its occurrence is relatively rare, contributing to a class imbalance in the dataset. A diverse array of factors influences the likelihood of a cyclone undergoing rapid intensification, further complicating the task for conventional machine learning models. In this paper, we evaluate deep learning, ensemble learning and data augmentation frameworks to detect cyclone rapid intensification based on wind intensity and spatial coordinates. We note that conventional data augmentation methods cannot be utilised for generating spatiotemporal patterns replicating cyclones that undergo rapid intensification. Therefore, our framework employs deep learning models to generate spatial coordinates and wind intensity that replicate cyclones to address the class imbalance problem of rapid intensification. We also use a deep learning model for the classification module within the data augmentation framework to differentiate between rapid and non-rapid intensification events during a cyclone. Our results show that data augmentation improves the results for rapid intensification detection in cyclones, and spatial coordinates play a critical role as input features to the given models. This paves the way for research in synthetic data generation for spatiotemporal data with extreme events. 

**Abstract (ZH)**: 热带气旋快速增强的深度学习检测方法研究 

---
# How Much To Guide: Revisiting Adaptive Guidance in Classifier-Free Guidance Text-to-Vision Diffusion Models 

**Title (ZH)**: 引导多少：重启无分类器引导的文字到视觉扩散模型中的自适应引导研究 

**Authors**: Huixuan Zhang, Junzhe Zhang, Xiaojun Wan  

**Link**: [PDF](https://arxiv.org/pdf/2506.08351)  

**Abstract**: With the rapid development of text-to-vision generation diffusion models, classifier-free guidance has emerged as the most prevalent method for conditioning. However, this approach inherently requires twice as many steps for model forwarding compared to unconditional generation, resulting in significantly higher costs. While previous study has introduced the concept of adaptive guidance, it lacks solid analysis and empirical results, making previous method unable to be applied to general diffusion models. In this work, we present another perspective of applying adaptive guidance and propose Step AG, which is a simple, universally applicable adaptive guidance strategy. Our evaluations focus on both image quality and image-text alignment. whose results indicate that restricting classifier-free guidance to the first several denoising steps is sufficient for generating high-quality, well-conditioned images, achieving an average speedup of 20% to 30%. Such improvement is consistent across different settings such as inference steps, and various models including video generation models, highlighting the superiority of our method. 

**Abstract (ZH)**: 基于步进的自适应指导：一种简单通用的方法及其应用 

---
# Highly Compressed Tokenizer Can Generate Without Training 

**Title (ZH)**: 高压缩词元化可以生成无需训练 

**Authors**: L. Lao Beyer, T. Li, X. Chen, S. Karaman, K. He  

**Link**: [PDF](https://arxiv.org/pdf/2506.08257)  

**Abstract**: Commonly used image tokenizers produce a 2D grid of spatially arranged tokens. In contrast, so-called 1D image tokenizers represent images as highly compressed one-dimensional sequences of as few as 32 discrete tokens. We find that the high degree of compression achieved by a 1D tokenizer with vector quantization enables image editing and generative capabilities through heuristic manipulation of tokens, demonstrating that even very crude manipulations -- such as copying and replacing tokens between latent representations of images -- enable fine-grained image editing by transferring appearance and semantic attributes. Motivated by the expressivity of the 1D tokenizer's latent space, we construct an image generation pipeline leveraging gradient-based test-time optimization of tokens with plug-and-play loss functions such as reconstruction or CLIP similarity. Our approach is demonstrated for inpainting and text-guided image editing use cases, and can generate diverse and realistic samples without requiring training of any generative model. 

**Abstract (ZH)**: 常用的图像分词器产生二维排列的空间分词网格。相比之下，所谓的1D图像分词器将图像表示为高度压缩的一维分词序列，最多仅包含32个离散分词。我们发现，通过向量量化实现的1D分词器的高度压缩程度，使其能够通过直觉操作分词实现图像编辑和生成能力，表明即使是非常粗糙的操作——如在图像的潜在表示之间复制和替换分词——也能通过转移外观和语义属性实现精细的图像编辑。受1D分词器潜在空间表达能力的启发，我们构建了一种图像生成管道，利用基于梯度的测试时分词优化及即插即用损失函数（如重建或CLIP相似性）进行操作。我们的方法在图像修复和文本引导的图像编辑应用中得到展示，并能在无需训练任何生成模型的情况下生成多样且真实的样本。 

---
# IGraSS: Learning to Identify Infrastructure Networks from Satellite Imagery by Iterative Graph-constrained Semantic Segmentation 

**Title (ZH)**: IGraSS：通过迭代图约束语义分割识别基础设施网络 

**Authors**: Oishee Bintey Hoque, Abhijin Adiga, Aniruddha Adiga, Siddharth Chaudhary, Madhav V. Marathe, S. S. Ravi, Kirti Rajagopalan, Amanda Wilson, Samarth Swarup  

**Link**: [PDF](https://arxiv.org/pdf/2506.08137)  

**Abstract**: Accurate canal network mapping is essential for water management, including irrigation planning and infrastructure maintenance. State-of-the-art semantic segmentation models for infrastructure mapping, such as roads, rely on large, well-annotated remote sensing datasets. However, incomplete or inadequate ground truth can hinder these learning approaches. Many infrastructure networks have graph-level properties such as reachability to a source (like canals) or connectivity (roads) that can be leveraged to improve these existing ground truth. This paper develops a novel iterative framework IGraSS, combining a semantic segmentation module-incorporating RGB and additional modalities (NDWI, DEM)-with a graph-based ground-truth refinement module. The segmentation module processes satellite imagery patches, while the refinement module operates on the entire data viewing the infrastructure network as a graph. Experiments show that IGraSS reduces unreachable canal segments from around 18% to 3%, and training with refined ground truth significantly improves canal identification. IGraSS serves as a robust framework for both refining noisy ground truth and mapping canal networks from remote sensing imagery. We also demonstrate the effectiveness and generalizability of IGraSS using road networks as an example, applying a different graph-theoretic constraint to complete road networks. 

**Abstract (ZH)**: 精确的沟渠网络测绘对于水资源管理，包括灌溉规划和基础设施维护至关重要。基于最新语义分割模型在基础设施测绘（如道路）中的应用依赖于大量且标注良好的遥感数据集。然而，缺乏或不充分的地面真实数据会阻碍这些学习方法。许多基础设施网络具有图级别特性，如可达性（如沟渠）或连接性（道路），这些特性可以利用以改进现有的地面真实数据。本文提出了一种新颖的迭代框架IGraSS，结合了一个语义分割模块（包含RGB和额外模态数据，如NDWI、DEM）与一个基于图的地面真实数据精炼模块。分割模块处理卫星图像片段，而精炼模块则在整个数据集上运行，将基础设施网络视为图。实验结果显示，IGraSS将未达沟渠段的比例从约18%降低到3%，使用精炼地面真实数据进行训练显著提高了沟渠识别效果。IGraSS作为一个鲁棒框架，可用于精炼噪音地面真实数据并从遥感图像中测绘沟渠网络。我们还通过使用道路网络作为示例，展示了IGraSS的有效性和普适性，并应用不同的图论约束来完成道路网络。 

---
