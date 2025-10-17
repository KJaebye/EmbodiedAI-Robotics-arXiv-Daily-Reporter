# Neural Implicit Flow Fields for Spatio-Temporal Motion Mapping 

**Title (ZH)**: 时空运动映射的神经隐式流场 

**Authors**: Yufei Zhu, Shih-Min Yang, Andrey Rudenko, Tomasz P. Kucner, Achim J. Lilienthal, Martin Magnusson  

**Link**: [PDF](https://arxiv.org/pdf/2510.14827)  

**Abstract**: Safe and efficient robot operation in complex human environments can benefit from good models of site-specific motion patterns. Maps of Dynamics (MoDs) provide such models by encoding statistical motion patterns in a map, but existing representations use discrete spatial sampling and typically require costly offline construction. We propose a continuous spatio-temporal MoD representation based on implicit neural functions that directly map coordinates to the parameters of a Semi-Wrapped Gaussian Mixture Model. This removes the need for discretization and imputation for unevenly sampled regions, enabling smooth generalization across both space and time. Evaluated on a large public dataset with long-term real-world people tracking data, our method achieves better accuracy of motion representation and smoother velocity distributions in sparse regions while still being computationally efficient, compared to available baselines. The proposed approach demonstrates a powerful and efficient way of modeling complex human motion patterns. 

**Abstract (ZH)**: 复杂人类环境中的安全高效机器人操作可以从特定场地的运动模式模型中受益。我们提出了一种基于隐式神经函数的连续时空MoD表示，该表示直接将坐标映射到半包卷混合模型的参数上，从而消除了离散化和不均匀采样区域的插补需求，能够在空间和时间上实现平滑泛化。在大型公共数据集上的评估结果表明，与现有基线方法相比，该方法在稀疏区域实现了更好的运动表示准确性和更平滑的速度分布，同时保持了计算效率。提出的 approach 展示了一种强大而高效的复杂人类运动模式建模方式。 

---
# A Generalized Placeability Metric for Model-Free Unified Pick-and-Place Reasoning 

**Title (ZH)**: 无模型统一取放推理的广义可放置性度量 

**Authors**: Benno Wingender, Nils Dengler, Rohit Menon, Sicong Pan, Maren Bennewitz  

**Link**: [PDF](https://arxiv.org/pdf/2510.14584)  

**Abstract**: To reliably pick and place unknown objects under real-world sensing noise remains a challenging task, as existing methods rely on strong object priors (e.g., CAD models), or planar-support assumptions, limiting generalization and unified reasoning between grasping and placing. In this work, we introduce a generalized placeability metric that evaluates placement poses directly from noisy point clouds, without any shape priors. The metric jointly scores stability, graspability, and clearance. From raw geometry, we extract the support surfaces of the object to generate diverse candidates for multi-orientation placement and sample contacts that satisfy collision and stability constraints. By conditioning grasp scores on each candidate placement, our proposed method enables model-free unified pick-and-place reasoning and selects grasp-place pairs that lead to stable, collision-free placements. On unseen real objects and non-planar object supports, our metric delivers CAD-comparable accuracy in predicting stability loss and generally produces more physically plausible placements than learning-based predictors. 

**Abstract (ZH)**: 在真实世界感知噪声下可靠地抓取和放置未知对象仍然是一个具有挑战性的任务，现有方法依赖于强大的物体先验（例如CAD模型）或平面支撑假设，限制了泛化能力和抓取与放置之间的统一推理。在本文中，我们引入了一个通用的放置度量，该度量直接从嘈杂的点云中评估放置姿态，而不依赖任何形式的形状先验。该度量联合评分稳定、可抓取性和避让性。从原始几何结构中，我们提取物体的支撑表面，生成多种朝向的放置候选，并采样满足碰撞和稳定性约束的接触点。通过在每个候选放置上条件化抓取评分，我们提出的方法实现了无需模型的统一抓取和放置推理，并选择能够导致稳定、无碰撞放置的抓取-放置配对。在未见过的真实对象和非平面支撑对象上，我们的度量在预测稳定性损失的准确性和生成更符合物理真实的放置方面与基于学习的预测器相比表现更佳。 

---
# Ponimator: Unfolding Interactive Pose for Versatile Human-human Interaction Animation 

**Title (ZH)**: Ponimator: 展开互动姿态以实现多样化的真人互动动画 

**Authors**: Shaowei Liu, Chuan Guo, Bing Zhou, Jian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14976)  

**Abstract**: Close-proximity human-human interactive poses convey rich contextual information about interaction dynamics. Given such poses, humans can intuitively infer the context and anticipate possible past and future dynamics, drawing on strong priors of human behavior. Inspired by this observation, we propose Ponimator, a simple framework anchored on proximal interactive poses for versatile interaction animation. Our training data consists of close-contact two-person poses and their surrounding temporal context from motion-capture interaction datasets. Leveraging interactive pose priors, Ponimator employs two conditional diffusion models: (1) a pose animator that uses the temporal prior to generate dynamic motion sequences from interactive poses, and (2) a pose generator that applies the spatial prior to synthesize interactive poses from a single pose, text, or both when interactive poses are unavailable. Collectively, Ponimator supports diverse tasks, including image-based interaction animation, reaction animation, and text-to-interaction synthesis, facilitating the transfer of interaction knowledge from high-quality mocap data to open-world scenarios. Empirical experiments across diverse datasets and applications demonstrate the universality of the pose prior and the effectiveness and robustness of our framework. 

**Abstract (ZH)**: 近距人体交互姿态承载丰富的交互动力学上下文信息。基于此类姿态，人类能够直观地推断出上下文并预判可能的过去和未来动态，依托强烈的人类行为先验。受到这一观察的启发，我们提出了Ponimator，一种基于近距交互姿态的多功能交互动画框架。我们的训练数据包括来自运动捕捉交互数据集的近距离接触两人姿态及其周围的时间上下文。利用交互姿态先验，Ponimator采用两个条件扩散模型：（1）一个姿态动画器，利用时间先验从交互姿态生成动态运动序列；（2）一个姿态生成器，应用空间先验从单个姿态、文本或两者合成交互姿态，当交互姿态不可用时。总体而言，Ponimator支持多种任务，包括基于图像的交互动画、反应动画以及文本到交互的合成，促进高质量运动捕捉数据中的交互知识向开放场景的转移。在不同数据集和应用领域的实证实验表明了姿态先验的通用性以及我们框架的有效性和鲁棒性。 

---
# Leveraging Cycle-Consistent Anchor Points for Self-Supervised RGB-D Registration 

**Title (ZH)**: 利用周期一致锚点进行自我监督的RGB-D配准 

**Authors**: Siddharth Tourani, Jayaram Reddy, Sarvesh Thakur, K Madhava Krishna, Muhammad Haris Khan, N Dinesh Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2510.14354)  

**Abstract**: With the rise in consumer depth cameras, a wealth of unlabeled RGB-D data has become available. This prompts the question of how to utilize this data for geometric reasoning of scenes. While many RGB-D registration meth- ods rely on geometric and feature-based similarity, we take a different approach. We use cycle-consistent keypoints as salient points to enforce spatial coherence constraints during matching, improving correspondence accuracy. Additionally, we introduce a novel pose block that combines a GRU recurrent unit with transformation synchronization, blending historical and multi-view data. Our approach surpasses previous self- supervised registration methods on ScanNet and 3DMatch, even outperforming some older supervised methods. We also integrate our components into existing methods, showing their effectiveness. 

**Abstract (ZH)**: 随着消费级深度相机的兴起，大量未标记的RGB-D数据变得可用。这促使我们思考如何利用这些数据来进行场景的几何推理。尽管许多RGB-D配准方法依赖于几何和特征相似性，我们采取了不同的方法。我们使用循环一致的关键点作为显著点，在匹配过程中施加空间连贯性约束，提高对应关系的准确性。此外，我们引入了一种新的姿态模块，该模块结合了GRU递归单元与变换同步，融合了历史和多视角数据。我们的方法在ScanNet和3DMatch上超过了之前的自监督配准方法，甚至优于一些较早的监督方法。我们还将我们的组件集成到现有方法中，展示了它们的有效性。 

---
# Coupled Diffusion Sampling for Training-Free Multi-View Image Editing 

**Title (ZH)**: 耦合扩散采样：无需训练的多视图图像编辑 

**Authors**: Hadi Alzayer, Yunzhi Zhang, Chen Geng, Jia-Bin Huang, Jiajun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14981)  

**Abstract**: We present an inference-time diffusion sampling method to perform multi-view consistent image editing using pre-trained 2D image editing models. These models can independently produce high-quality edits for each image in a set of multi-view images of a 3D scene or object, but they do not maintain consistency across views. Existing approaches typically address this by optimizing over explicit 3D representations, but they suffer from a lengthy optimization process and instability under sparse view settings. We propose an implicit 3D regularization approach by constraining the generated 2D image sequences to adhere to a pre-trained multi-view image distribution. This is achieved through coupled diffusion sampling, a simple diffusion sampling technique that concurrently samples two trajectories from both a multi-view image distribution and a 2D edited image distribution, using a coupling term to enforce the multi-view consistency among the generated images. We validate the effectiveness and generality of this framework on three distinct multi-view image editing tasks, demonstrating its applicability across various model architectures and highlighting its potential as a general solution for multi-view consistent editing. 

**Abstract (ZH)**: 我们提出了一种推理时的扩散采样方法，使用预训练的2D图像编辑模型在多视角图像中进行一致的图像编辑。 

---
# WithAnyone: Towards Controllable and ID Consistent Image Generation 

**Title (ZH)**: WithAnyone: 向可控且ID一致的图像生成努力 

**Authors**: Hengyuan Xu, Wei Cheng, Peng Xing, Yixiao Fang, Shuhan Wu, Rui Wang, Xianfang Zeng, Daxin Jiang, Gang Yu, Xingjun Ma, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14975)  

**Abstract**: Identity-consistent generation has become an important focus in text-to-image research, with recent models achieving notable success in producing images aligned with a reference identity. Yet, the scarcity of large-scale paired datasets containing multiple images of the same individual forces most approaches to adopt reconstruction-based training. This reliance often leads to a failure mode we term copy-paste, where the model directly replicates the reference face rather than preserving identity across natural variations in pose, expression, or lighting. Such over-similarity undermines controllability and limits the expressive power of generation. To address these limitations, we (1) construct a large-scale paired dataset MultiID-2M, tailored for multi-person scenarios, providing diverse references for each identity; (2) introduce a benchmark that quantifies both copy-paste artifacts and the trade-off between identity fidelity and variation; and (3) propose a novel training paradigm with a contrastive identity loss that leverages paired data to balance fidelity with diversity. These contributions culminate in WithAnyone, a diffusion-based model that effectively mitigates copy-paste while preserving high identity similarity. Extensive qualitative and quantitative experiments demonstrate that WithAnyone significantly reduces copy-paste artifacts, improves controllability over pose and expression, and maintains strong perceptual quality. User studies further validate that our method achieves high identity fidelity while enabling expressive controllable generation. 

**Abstract (ZH)**: 身份一致生成已成为文本到图像研究的重要关注点，近期的模型在生成与参考身份一致的图像方面取得了显著成果。然而，缺乏包含同一个体多张图片的大规模配对数据集促使大多数方法采用基于重构的训练。这种依赖往往导致我们称之为复制粘贴的失败模式，即模型直接复制参考面部而不是在姿态、表情或光照的自然变化中保持身份一致性。这种过度相似性削弱了可控性并限制了生成的表达力。为了应对这些局限性，我们（1）构建了一个适用于多人大场景的大型配对数据集MultiID-2M，为每个身份提供多样化的参考；（2）引入了一个基准，量化复制粘贴的伪影以及身份忠实度与多样性之间的权衡；（3）提出了一种新的训练范式，利用配对数据引入对比身份损失，平衡忠实度与多样性。这些贡献导致了WithAnyone模型的提出，该模型基于扩散模型，能够有效缓解复制粘贴问题，同时保持高身份相似性。大量定性和定量实验表明，WithAnyone显著减少了复制粘贴伪影，提高了对姿态和表情的可控性，并保持了强烈的感知质量。用户研究进一步验证了我们的方法在保持高身份忠实度的同时实现了表达性可控生成。 

---
# pi-Flow: Policy-Based Few-Step Generation via Imitation Distillation 

**Title (ZH)**: pi-Flow: 基于策略的 Few-Step 生成通过模仿提炼 

**Authors**: Hansheng Chen, Kai Zhang, Hao Tan, Leonidas Guibas, Gordon Wetzstein, Sai Bi  

**Link**: [PDF](https://arxiv.org/pdf/2510.14974)  

**Abstract**: Few-step diffusion or flow-based generative models typically distill a velocity-predicting teacher into a student that predicts a shortcut towards denoised data. This format mismatch has led to complex distillation procedures that often suffer from a quality-diversity trade-off. To address this, we propose policy-based flow models ($\pi$-Flow). $\pi$-Flow modifies the output layer of a student flow model to predict a network-free policy at one timestep. The policy then produces dynamic flow velocities at future substeps with negligible overhead, enabling fast and accurate ODE integration on these substeps without extra network evaluations. To match the policy's ODE trajectory to the teacher's, we introduce a novel imitation distillation approach, which matches the policy's velocity to the teacher's along the policy's trajectory using a standard $\ell_2$ flow matching loss. By simply mimicking the teacher's behavior, $\pi$-Flow enables stable and scalable training and avoids the quality-diversity trade-off. On ImageNet 256$^2$, it attains a 1-NFE FID of 2.85, outperforming MeanFlow of the same DiT architecture. On FLUX.1-12B and Qwen-Image-20B at 4 NFEs, $\pi$-Flow achieves substantially better diversity than state-of-the-art few-step methods, while maintaining teacher-level quality. 

**Abstract (ZH)**: 基于策略的流模型（$\pi$-Flow） 

---
# C4D: 4D Made from 3D through Dual Correspondences 

**Title (ZH)**: C4D：通过双对应关系从3D生成4D 

**Authors**: Shizun Wang, Zhenxiang Jiang, Xingyi Yang, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14960)  

**Abstract**: Recovering 4D from monocular video, which jointly estimates dynamic geometry and camera poses, is an inevitably challenging problem. While recent pointmap-based 3D reconstruction methods (e.g., DUSt3R) have made great progress in reconstructing static scenes, directly applying them to dynamic scenes leads to inaccurate results. This discrepancy arises because moving objects violate multi-view geometric constraints, disrupting the reconstruction. To address this, we introduce C4D, a framework that leverages temporal Correspondences to extend existing 3D reconstruction formulation to 4D. Specifically, apart from predicting pointmaps, C4D captures two types of correspondences: short-term optical flow and long-term point tracking. We train a dynamic-aware point tracker that provides additional mobility information, facilitating the estimation of motion masks to separate moving elements from the static background, thus offering more reliable guidance for dynamic scenes. Furthermore, we introduce a set of dynamic scene optimization objectives to recover per-frame 3D geometry and camera parameters. Simultaneously, the correspondences lift 2D trajectories into smooth 3D trajectories, enabling fully integrated 4D reconstruction. Experiments show that our framework achieves complete 4D recovery and demonstrates strong performance across multiple downstream tasks, including depth estimation, camera pose estimation, and point tracking. Project Page: this https URL 

**Abstract (ZH)**: 从单目视频恢复4D：一种联合估计动态几何和相机姿态的框架，是不可避免的挑战性问题。虽然最近基于点图的3D重建方法（例如DUSt3R）在重建静态场景方面取得了巨大进步，但直接将其应用于动态场景会导致不准确的结果。这种差异源于移动物体违反了多视图几何约束，干扰了重建。为解决这一问题，我们提出C4D框架，其利用时间对应关系将现有的3D重建公式扩展到4D。具体而言，C4D不仅预测点图，还捕获两种类型的对应关系：短期光学流和长期点跟踪。我们训练了一个动态感知的点跟踪器，提供额外的移动性信息，促进运动掩码的估计，以分离移动元素和静态背景，从而为动态场景提供更多可靠的指导。此外，我们引入了一组动态场景优化目标，以恢复每帧的3D几何和相机参数。同时，对应关系将2D轨迹提升为平滑的3D轨迹，实现全面集成的4D重建。实验结果显示，我们的框架实现了完整的4D恢复，并在多个下游任务中表现出强大的性能，包括深度估计、相机姿态估计和点跟踪。项目页面：this https URL 

---
# RealDPO: Real or Not Real, that is the Preference 

**Title (ZH)**: RealDPO: 实或虚，偏好决定 

**Authors**: Guo Cheng, Danni Yang, Ziqi Huang, Jianlou Si, Chenyang Si, Ziwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14955)  

**Abstract**: Video generative models have recently achieved notable advancements in synthesis quality. However, generating complex motions remains a critical challenge, as existing models often struggle to produce natural, smooth, and contextually consistent movements. This gap between generated and real-world motions limits their practical applicability. To address this issue, we introduce RealDPO, a novel alignment paradigm that leverages real-world data as positive samples for preference learning, enabling more accurate motion synthesis. Unlike traditional supervised fine-tuning (SFT), which offers limited corrective feedback, RealDPO employs Direct Preference Optimization (DPO) with a tailored loss function to enhance motion realism. By contrasting real-world videos with erroneous model outputs, RealDPO enables iterative self-correction, progressively refining motion quality. To support post-training in complex motion synthesis, we propose RealAction-5K, a curated dataset of high-quality videos capturing human daily activities with rich and precise motion details. Extensive experiments demonstrate that RealDPO significantly improves video quality, text alignment, and motion realism compared to state-of-the-art models and existing preference optimization techniques. 

**Abstract (ZH)**: 视频生成模型最近在合成质量方面取得了显著进步。然而，生成复杂的运动仍然是一个关键挑战，因为现有的模型往往难以产生自然、流畅且上下文一致的运动。生成的运动与真实世界运动之间的差距限制了其实际应用。为了解决这一问题，我们引入了RealDPO，这是一种新颖的对齐范式，通过使用真实世界数据作为偏好学习的正样本，以实现更准确的运动合成。与传统的监督微调（SFT）相比，后者提供的纠正反馈有限，RealDPO利用直接偏好优化（DPO）和定制的损失函数来增强运动的真实感。通过将真实世界视频与模型的错误输出进行对比，RealDPO能够实现迭代自我纠正，逐步提高运动质量。为了支持复杂运动合成的后训练，我们提出了RealAction-5K数据集，这是一个精心编制的高质量视频数据集，捕捉了人类日常活动中的丰富和精细的运动细节。大量实验表明，与最先进的模型和现有的偏好优化技术相比，RealDPO显著提高了视频质量、文本对齐和运动的真实感。 

---
# MaskCaptioner : Learning to Jointly Segment and Caption Object Trajectories in Videos 

**Title (ZH)**: MaskCaptioner：学习联合分割和描述视频中对象轨迹 

**Authors**: Gabriel Fiastre, Antoine Yang, Cordelia Schmid  

**Link**: [PDF](https://arxiv.org/pdf/2510.14904)  

**Abstract**: Dense Video Object Captioning (DVOC) is the task of jointly detecting, tracking, and captioning object trajectories in a video, requiring the ability to understand spatio-temporal details and describe them in natural language. Due to the complexity of the task and the high cost associated with manual annotation, previous approaches resort to disjoint training strategies, potentially leading to suboptimal performance. To circumvent this issue, we propose to generate captions about spatio-temporally localized entities leveraging a state-of-the-art VLM. By extending the LVIS and LV-VIS datasets with our synthetic captions (LVISCap and LV-VISCap), we train MaskCaptioner, an end-to-end model capable of jointly detecting, segmenting, tracking and captioning object trajectories. Moreover, with pretraining on LVISCap and LV-VISCap, MaskCaptioner achieves state-of-the-art DVOC results on three existing benchmarks, VidSTG, VLN and BenSMOT. The datasets and code are available at this https URL. 

**Abstract (ZH)**: 稠密视频对象描述（DVOC）是联合检测、跟踪和描述视频中对象轨迹的任务，要求能够理解时空细节并用自然语言描述。由于任务的复杂性和手动标注的高成本，先前的方法采用分离训练策略，可能导致性能不佳。为解决这一问题，我们提出利用先进的视觉语言模型生成时空局部化实体的描述。通过将LVIS和LV-VIS数据集扩展为包含我们合成的描述（LVISCap和LV-VISCap），我们训练了一个端到端模型MaskCaptioner，该模型能够联合检测、分割、跟踪和描述对象轨迹。此外，通过在LVISCap和LV-VISCap上的预训练，MaskCaptioner在三个现有基准VidSTG、VLN和BenSMOT上达到了最先进的DVOC结果。数据集和代码可在以下链接获取。 

---
# Inpainting the Red Planet: Diffusion Models for the Reconstruction of Martian Environments in Virtual Reality 

**Title (ZH)**: 虚拟现实中火星环境的修复：扩散模型在重建火星环境中的应用 

**Authors**: Giuseppe Lorenzo Catalano, Agata Marta Soccini  

**Link**: [PDF](https://arxiv.org/pdf/2510.14765)  

**Abstract**: Space exploration increasingly relies on Virtual Reality for several tasks, such as mission planning, multidisciplinary scientific analysis, and astronaut training. A key factor for the reliability of the simulations is having accurate 3D representations of planetary terrains. Extraterrestrial heightmaps derived from satellite imagery often contain missing values due to acquisition and transmission constraints. Mars is among the most studied planets beyond Earth, and its extensive terrain datasets make the Martian surface reconstruction a valuable task, although many areas remain unmapped. Deep learning algorithms can support void-filling tasks; however, whereas Earth's comprehensive datasets enables the use of conditional methods, such approaches cannot be applied to Mars. Current approaches rely on simpler interpolation techniques which, however, often fail to preserve geometric coherence. In this work, we propose a method for reconstructing the surface of Mars based on an unconditional diffusion model. Training was conducted on an augmented dataset of 12000 Martian heightmaps derived from NASA's HiRISE survey. A non-homogeneous rescaling strategy captures terrain features across multiple scales before resizing to a fixed 128x128 model resolution. We compared our method against established void-filling and inpainting techniques, including Inverse Distance Weighting, kriging, and Navier-Stokes algorithm, on an evaluation set of 1000 samples. Results show that our approach consistently outperforms these methods in terms of reconstruction accuracy (4-15% on RMSE) and perceptual similarity (29-81% on LPIPS) with the original data. 

**Abstract (ZH)**: 外太空探索越来越多地依赖虚拟现实进行任务规划、跨学科科学分析和宇航员训练。高度图的准确三维表示是模拟可靠性的关键因素。由于获取和传输约束，从卫星图像派生的外星球高度图经常包含缺失值。火星是除地球外研究最多的行星之一，其庞大的地形数据集使火星表面重建成为一个有价值的任务，尽管许多地区尚未被测绘。深度学习算法可以支持空值填充任务；然而，由于地球数据的全面性，可以在其中应用条件方法，而这些方法在火星上无法应用。当前的方法依赖于更简单的插值技术，但这些技术往往无法保持几何连贯性。在这项工作中，我们提出了一种基于无条件扩散模型的火星表面重建方法。训练数据集包含12000个由NASA HiRISE调查派生的高度图，并进行了扩充。非均匀缩放策略在缩放至固定128x128模型分辨率之前捕捉跨多个尺度的地形特征。我们使用1000个样本的评估集将我们的方法与已建立的空值填充和修复技术（包括距离加权法、克里金法和Navier-Stokes算法）进行了比较。结果显示，在均方根误差和LPIPS感知相似度方面，我们的方法在重建准确性上始终优于这些方法（4-15%的均方根误差改善和29-81%的LPIPS感知相似度提升）。 

---
# Camera Movement Classification in Historical Footage: A Comparative Study of Deep Video Models 

**Title (ZH)**: 历史影像中的摄像头运动分类：深度视频模型的比较研究 

**Authors**: Tingyu Lin, Armin Dadras, Florian Kleber, Robert Sablatnig  

**Link**: [PDF](https://arxiv.org/pdf/2510.14713)  

**Abstract**: Camera movement conveys spatial and narrative information essential for understanding video content. While recent camera movement classification (CMC) methods perform well on modern datasets, their generalization to historical footage remains unexplored. This paper presents the first systematic evaluation of deep video CMC models on archival film material. We summarize representative methods and datasets, highlighting differences in model design and label definitions. Five standard video classification models are assessed on the HISTORIAN dataset, which includes expert-annotated World War II footage. The best-performing model, Video Swin Transformer, achieves 80.25% accuracy, showing strong convergence despite limited training data. Our findings highlight the challenges and potential of adapting existing models to low-quality video and motivate future work combining diverse input modalities and temporal architectures. 

**Abstract (ZH)**: 相机运动传递了理解视频内容所需的空问和叙述信息。尽管近期的相机运动分类方法在现代数据集上表现良好，但其对历史片段的泛化能力尚未被探索。本文首次系统评估了深度视频相机运动分类模型在档案电影材料上的表现。我们总结了代表性的方法和数据集，强调了模型设计和标签定义的差异。五种标准的视频分类模型在包含专家标注的二战 footage 的 HISTORIAN 数据集上进行了评估。性能最好的 Video Swin Transformer 模型达到了 80.25% 的准确率，尽管训练数据有限，但仍表现出较强的收敛性。我们的研究结果凸显了现有模型适应低质量视频的挑战和潜力，并激发了未来结合多种输入模态和时空架构的研究工作。 

---
# Where are the Whales: A Human-in-the-loop Detection Method for Identifying Whales in High-resolution Satellite Imagery 

**Title (ZH)**: Whale何在：一种基于人的回路高分辨率卫星图像中识别鲸鱼的检测方法 

**Authors**: Caleb Robinson, Kimberly T. Goetz, Christin B. Khan, Meredith Sackett, Kathleen Leonard, Rahul Dodhia, Juan M. Lavista Ferres  

**Link**: [PDF](https://arxiv.org/pdf/2510.14709)  

**Abstract**: Effective monitoring of whale populations is critical for conservation, but traditional survey methods are expensive and difficult to scale. While prior work has shown that whales can be identified in very high-resolution (VHR) satellite imagery, large-scale automated detection remains challenging due to a lack of annotated imagery, variability in image quality and environmental conditions, and the cost of building robust machine learning pipelines over massive remote sensing archives. We present a semi-automated approach for surfacing possible whale detections in VHR imagery using a statistical anomaly detection method that flags spatial outliers, i.e. "interesting points". We pair this detector with a web-based labeling interface designed to enable experts to quickly annotate the interesting points. We evaluate our system on three benchmark scenes with known whale annotations and achieve recalls of 90.3% to 96.4%, while reducing the area requiring expert inspection by up to 99.8% -- from over 1,000 sq km to less than 2 sq km in some cases. Our method does not rely on labeled training data and offers a scalable first step toward future machine-assisted marine mammal monitoring from space. We have open sourced this pipeline at this https URL. 

**Abstract (ZH)**: 有效的鲸群监测对于保护至关重要，但传统调查方法成本高昂且难以扩展。尽管先前的工作证明了可以在非常高分辨率（VHR）卫星图像中识别鲸鱼，但由于缺乏标注图像、图像质量和环境条件的变异性以及构建 robust 机器学习管道的高成本，大规模自动化检测仍具有挑战性。我们提出了一种半自动化方法，通过统计异常检测方法在VHR图像中 surface 可能的鲸鱼检测结果，该方法标记空间异常点，即“有趣点”。我们使用了一个基于Web的标注界面，以使专家能够快速标注这些有趣点。我们在三个具有已知鲸鱼标注的基准场景上评估了该系统，召回率达到了 90.3% 至 96.4%，并且在某些情况下将需要专家检查的区域减少了99.8%，从超过1,000平方公里减少到不到2平方公里。我们的方法不依赖于标注训练数据，并为未来基于机器辅助的空间内海哺乳动物监测提供了一步可扩展的步骤。我们已在以下网址开源了此管道：this https URL。 

---
# Galaxy Morphology Classification with Counterfactual Explanation 

**Title (ZH)**: 银河系形态分类与反事实解释 

**Authors**: Zhuo Cao, Lena Krieger, Hanno Scharr, Ira Assent  

**Link**: [PDF](https://arxiv.org/pdf/2510.14655)  

**Abstract**: Galaxy morphologies play an essential role in the study of the evolution of galaxies. The determination of morphologies is laborious for a large amount of data giving rise to machine learning-based approaches. Unfortunately, most of these approaches offer no insight into how the model works and make the results difficult to understand and explain. We here propose to extend a classical encoder-decoder architecture with invertible flow, allowing us to not only obtain a good predictive performance but also provide additional information about the decision process with counterfactual explanations. 

**Abstract (ZH)**: galaxies的形态在研究星系演化中发挥着重要作用。形态的确定对于大量数据来说是劳动密集型的工作，因此导致了基于机器学习的方法。然而，大多数这些方法未能提供模型工作原理的见解，使结果难以理解和解释。我们提出将经典编码解码架构扩展为包含可逆流的架构，从而不仅获得良好的预测性能，还能通过事实假设解释提供决策过程的附加信息。 

---
# In-Context Learning with Unpaired Clips for Instruction-based Video Editing 

**Title (ZH)**: 基于指令的视频编辑中未配对剪辑的上下文学习 

**Authors**: Xinyao Liao, Xianfang Zeng, Ziye Song, Zhoujie Fu, Gang Yu, Guosheng Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.14648)  

**Abstract**: Despite the rapid progress of instruction-based image editing, its extension to video remains underexplored, primarily due to the prohibitive cost and complexity of constructing large-scale paired video editing datasets. To address this challenge, we introduce a low-cost pretraining strategy for instruction-based video editing that leverages in-context learning from unpaired video clips. We show that pretraining a foundation video generation model with this strategy endows it with general editing capabilities, such as adding, replacing, or deleting operations, according to input editing instructions. The pretrained model can then be efficiently refined with a small amount of high-quality paired editing data. Built upon HunyuanVideoT2V, our framework first pretrains on approximately 1M real video clips to learn basic editing concepts, and subsequently fine-tunes on fewer than 150k curated editing pairs to extend more editing tasks and improve the editing quality. Comparative experiments show that our method surpasses existing instruction-based video editing approaches in both instruction alignment and visual fidelity, achieving a 12\% improvement in editing instruction following and a 15\% improvement in editing quality. 

**Abstract (ZH)**: 尽管基于指令的图像编辑取得了 rapid progress，其在视频编辑领域的扩展仍鲜有探索，主要原因是构建大规模配对视频编辑数据集的成本高且复杂。为解决这一挑战，我们提出了一种基于指令的低成本预训练策略，利用非配对视频片段的上下文学习进行训练。我们展示了使用该策略预训练基础视频生成模型，使其获得根据输入编辑指令执行添加、替换或删除等通用编辑能力。预训练后的模型可以使用少量高质量的配对编辑数据进行高效调优。我们的框架基于HunyuanVideoT2V，首先在约100万真实视频片段上进行预训练以学习基本编辑概念，然后使用不到15万精心挑选的编辑配对进行微调，以扩展更多编辑任务并提高编辑质量。对比实验显示，我们的方法在指令对齐和视觉保真度方面均优于现有基于指令的视频编辑方法，在编辑指令跟随上提升了12%，在编辑质量上提升了15%。 

---
# STANCE: Motion Coherent Video Generation Via Sparse-to-Dense Anchored Encoding 

**Title (ZH)**: stance: 通过稀疏到稠密锚定编码实现运动连贯的视频生成 

**Authors**: Zhifei Chen, Tianshuo Xu, Leyi Wu, Luozhou Wang, Dongyu Yan, Zihan You, Wenting Luo, Guo Zhang, Yingcong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.14588)  

**Abstract**: Video generation has recently made striking visual progress, but maintaining coherent object motion and interactions remains difficult. We trace two practical bottlenecks: (i) human-provided motion hints (e.g., small 2D maps) often collapse to too few effective tokens after encoding, weakening guidance; and (ii) optimizing for appearance and motion in a single head can favor texture over temporal consistency. We present STANCE, an image-to-video framework that addresses both issues with two simple components. First, we introduce Instance Cues -- a pixel-aligned control signal that turns sparse, user-editable hints into a dense 2.5D (camera-relative) motion field by averaging per-instance flow and augmenting with monocular depth over the instance mask. This reduces depth ambiguity compared to 2D arrow inputs while remaining easy to use. Second, we preserve the salience of these cues in token space with Dense RoPE, which tags a small set of motion tokens (anchored on the first frame) with spatial-addressable rotary embeddings. Paired with joint RGB \(+\) auxiliary-map prediction (segmentation or depth), our model anchors structure while RGB handles appearance, stabilizing optimization and improving temporal coherence without requiring per-frame trajectory scripts. 

**Abstract (ZH)**: 基于实例的线索和密集RoPE的视频生成框架：解决连贯对象运动和交互的问题 

---
# Beat Detection as Object Detection 

**Title (ZH)**: 鼓点检测作为对象检测 

**Authors**: Jaehoon Ahn, Moon-Ryul Jung  

**Link**: [PDF](https://arxiv.org/pdf/2510.14391)  

**Abstract**: Recent beat and downbeat tracking models (e.g., RNNs, TCNs, Transformers) output frame-level activations. We propose reframing this task as object detection, where beats and downbeats are modeled as temporal "objects." Adapting the FCOS detector from computer vision to 1D audio, we replace its original backbone with WaveBeat's temporal feature extractor and add a Feature Pyramid Network to capture multi-scale temporal patterns. The model predicts overlapping beat/downbeat intervals with confidence scores, followed by non-maximum suppression (NMS) to select final predictions. This NMS step serves a similar role to DBNs in traditional trackers, but is simpler and less heuristic. Evaluated on standard music datasets, our approach achieves competitive results, showing that object detection techniques can effectively model musical beats with minimal adaptation. 

**Abstract (ZH)**: 最近的打击声和重拍跟踪模型（如RNNs、TCNs、Transformers）输出帧级激活。我们提出了将此任务重新定义为对象检测，其中打击声和重拍被 modeling 为时间上的“对象”。将来自于计算机视觉领域的 FCOS 检测器适应至 1D 音频，我们用 WaveBeat 的时间特征提取器替换其原始骨干，并添加 Feature Pyramid Network 以捕捉多层次的时间模式。该模型预测具有置信分数的重叠的打击声/重拍区间，随后通过非最大抑制（NMS）选择最终预测。该 NMS 步骤在传统跟踪器中类似于 DBNs 的作用，但更为简单且减少了几何硬性约束。在标准音乐数据集上评估，我们的方法取得竞争性的结果，表明对象检测技术在少量调整的情况下能够有效建模音乐节奏。 

---
# SUM-AgriVLN: Spatial Understanding Memory for Agricultural Vision-and-Language Navigation 

**Title (ZH)**: SUM-AgriVLN: 空间理解记忆在农业视觉-语言导航中的应用 

**Authors**: Xiaobei Zhao, Xingqi Lyu, Xiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.14357)  

**Abstract**: Agricultural robots are emerging as powerful assistants across a wide range of agricultural tasks, nevertheless, still heavily rely on manual operation or fixed rail systems for movement. The AgriVLN method and the A2A benchmark pioneeringly extend Vision-and-Language Navigation (VLN) to the agricultural domain, enabling robots to navigate to the target positions following the natural language instructions. In practical agricultural scenarios, navigation instructions often repeatedly occur, yet AgriVLN treat each instruction as an independent episode, overlooking the potential of past experiences to provide spatial context for subsequent ones. To bridge this gap, we propose the method of Spatial Understanding Memory for Agricultural Vision-and-Language Navigation (SUM-AgriVLN), in which the SUM module employs spatial understanding and save spatial memory through 3D reconstruction and representation. When evaluated on the A2A benchmark, our SUM-AgriVLN effectively improves Success Rate from 0.47 to 0.54 with slight sacrifice on Navigation Error from 2.91m to 2.93m, demonstrating the state-of-the-art performance in the agricultural domain. Code: this https URL. 

**Abstract (ZH)**: 农业机器人正在广泛农业生产任务中崭露头角，但仍主要依赖手动操作或固定轨道系统进行移动。AgriVLN方法和A2A基准首次将视觉与语言导航（VLN）扩展到农业领域，使机器人能够根据自然语言指令导航至目标位置。在实际农业生产场景中，导航指令常常重复出现，但AgriVLN将每条指令视为独立的 episode，忽视了过往经验在提供后续指令空间上下文方面的潜力。为解决这一问题，我们提出了农业视觉与语言导航的空间理解记忆方法（SUM-AgriVLN），其中SUM模块通过三维重建和表示来实现空间理解和保存空间记忆。在A2A基准上进行评估时，我们的SUM-AgriVLN有效提高了成功率达到0.54，同时略微增加了导航误差从2.91米到2.93米，证明了在农业领域的先进性能。代码：这个 https URL。 

---
# Reinforcement Learning for Unsupervised Domain Adaptation in Spatio-Temporal Echocardiography Segmentation 

**Title (ZH)**: 时空超声心肌分割的无监督领域适应强化学习 

**Authors**: Arnaud Judge, Nicolas Duchateau, Thierry Judge, Roman A. Sandler, Joseph Z. Sokol, Christian Desrosiers, Olivier Bernard, Pierre-Marc Jodoin  

**Link**: [PDF](https://arxiv.org/pdf/2510.14244)  

**Abstract**: Domain adaptation methods aim to bridge the gap between datasets by enabling knowledge transfer across domains, reducing the need for additional expert annotations. However, many approaches struggle with reliability in the target domain, an issue particularly critical in medical image segmentation, where accuracy and anatomical validity are essential. This challenge is further exacerbated in spatio-temporal data, where the lack of temporal consistency can significantly degrade segmentation quality, and particularly in echocardiography, where the presence of artifacts and noise can further hinder segmentation performance. To address these issues, we present RL4Seg3D, an unsupervised domain adaptation framework for 2D + time echocardiography segmentation. RL4Seg3D integrates novel reward functions and a fusion scheme to enhance key landmark precision in its segmentations while processing full-sized input videos. By leveraging reinforcement learning for image segmentation, our approach improves accuracy, anatomical validity, and temporal consistency while also providing, as a beneficial side effect, a robust uncertainty estimator, which can be used at test time to further enhance segmentation performance. We demonstrate the effectiveness of our framework on over 30,000 echocardiographic videos, showing that it outperforms standard domain adaptation techniques without the need for any labels on the target domain. Code is available at this https URL. 

**Abstract (ZH)**: 基于RL的无监督3D心超时空分割领域适应方法_rl4seg3D 

---
# Virtually Being: Customizing Camera-Controllable Video Diffusion Models with Multi-View Performance Captures 

**Title (ZH)**: 虚拟存在：基于多视角性能捕捉的自定义摄像机可控视频扩散模型 

**Authors**: Yuancheng Xu, Wenqi Xian, Li Ma, Julien Philip, Ahmet Levent Taşel, Yiwei Zhao, Ryan Burgert, Mingming He, Oliver Hermann, Oliver Pilarski, Rahul Garg, Paul Debevec, Ning Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14179)  

**Abstract**: We introduce a framework that enables both multi-view character consistency and 3D camera control in video diffusion models through a novel customization data pipeline. We train the character consistency component with recorded volumetric capture performances re-rendered with diverse camera trajectories via 4D Gaussian Splatting (4DGS), lighting variability obtained with a video relighting model. We fine-tune state-of-the-art open-source video diffusion models on this data to provide strong multi-view identity preservation, precise camera control, and lighting adaptability. Our framework also supports core capabilities for virtual production, including multi-subject generation using two approaches: joint training and noise blending, the latter enabling efficient composition of independently customized models at inference time; it also achieves scene and real-life video customization as well as control over motion and spatial layout during customization. Extensive experiments show improved video quality, higher personalization accuracy, and enhanced camera control and lighting adaptability, advancing the integration of video generation into virtual production. Our project page is available at: this https URL. 

**Abstract (ZH)**: 我们提出了一种框架，通过一种新颖的自定义数据管道，实现了视频扩散模型中的多视图角色一致性与3D相机控制。我们使用4D高斯点图（4DGS）重渲染记录的体三维捕捉表演，并结合视频重光照模型获得的光照变化，训练角色一致性组件。我们在这些数据上对最先进的开源视频扩散模型进行微调，以提供强大的多视图身份保真、精确的相机控制和光照适应性。该框架还支持虚拟生产的核心能力，包括使用两种方法（联合训练和噪声融合）进行多主体生成，后者在推理时能高效地组合独立定制的模型；同时实现了场景和现实生活视频的自定义，以及在自定义过程中对动作和空间布局的控制。大量实验表明，视频质量得到了提高，个性化精度更高，相机控制和光照适应性增强，推动了视频生成与虚拟生产的融合。我们的项目页面可在以下链接访问：this https URL。 

---
# Efficient Few-Shot Learning in Remote Sensing: Fusing Vision and Vision-Language Models 

**Title (ZH)**: 遥感中的高效少-shot学习：视觉与视觉语言模型融合 

**Authors**: Jia Yun Chua, Argyrios Zolotas, Miguel Arana-Catania  

**Link**: [PDF](https://arxiv.org/pdf/2510.13993)  

**Abstract**: Remote sensing has become a vital tool across sectors such as urban planning, environmental monitoring, and disaster response. While the volume of data generated has increased significantly, traditional vision models are often constrained by the requirement for extensive domain-specific labelled data and their limited ability to understand the context within complex environments. Vision Language Models offer a complementary approach by integrating visual and textual data; however, their application to remote sensing remains underexplored, particularly given their generalist nature. This work investigates the combination of vision models and VLMs to enhance image analysis in remote sensing, with a focus on aircraft detection and scene understanding. The integration of YOLO with VLMs such as LLaVA, ChatGPT, and Gemini aims to achieve more accurate and contextually aware image interpretation. Performance is evaluated on both labelled and unlabelled remote sensing data, as well as degraded image scenarios which are crucial for remote sensing. The findings show an average MAE improvement of 48.46% across models in the accuracy of aircraft detection and counting, especially in challenging conditions, in both raw and degraded scenarios. A 6.17% improvement in CLIPScore for comprehensive understanding of remote sensing images is obtained. The proposed approach combining traditional vision models and VLMs paves the way for more advanced and efficient remote sensing image analysis, especially in few-shot learning scenarios. 

**Abstract (ZH)**: 遥感已成为城市规划、环境监测和灾害响应等领域的重要工具。虽然生成的数据量显著增加，但传统视觉模型往往受限于需要大量专用领域标注数据，并且在理解复杂环境中的上下文方面能力有限。视觉语言模型通过整合视觉和文本数据提供了互补的方法；然而，它们在遥感领域的应用尚未得到充分探索，特别是考虑到它们的通用性。本研究调查了将视觉模型与VLMs结合以增强遥感图像分析的方法，重点在于飞机检测和场景理解。通过将YOLO与LLaVA、ChatGPT、Gemini等VLMs集成，旨在实现更准确和上下文感知的图像解释。性能在标记和未标记的遥感数据以及降质图像场景中进行了评估，后者对于遥感至关重要。研究结果表明，在飞机检测和计数的准确性上，模型平均提高了48.46%，尤其是在挑战性条件下，在原始和降质场景中尤为明显。遥感图像综合理解的CLIPScore提高了6.17%。结合传统视觉模型和VLMs的方法为更高级和高效的遥感图像分析铺平了道路，特别是在少量样本学习场景中。 

---
# Dual-attention ResNet outperforms transformers in HER2 prediction on DCE-MRI 

**Title (ZH)**: Dual-attention ResNet在DCE-MRI的HER2预测中优于 Transformers 

**Authors**: Naomi Fridman, Anat Goldstein  

**Link**: [PDF](https://arxiv.org/pdf/2510.13897)  

**Abstract**: Breast cancer is the most diagnosed cancer in women, with HER2 status critically guiding treatment decisions. Noninvasive prediction of HER2 status from dynamic contrast-enhanced MRI (DCE-MRI) could streamline diagnostics and reduce reliance on biopsy. However, preprocessing high-dynamic-range DCE-MRI into standardized 8-bit RGB format for pretrained neural networks is nontrivial, and normalization strategy significantly affects model performance. We benchmarked intensity normalization strategies using a Triple-Head Dual-Attention ResNet that processes RGB-fused temporal sequences from three DCE phases. Trained on a multicenter cohort (n=1,149) from the I-SPY trials and externally validated on BreastDCEDL_AMBL (n=43 lesions), our model outperformed transformer-based architectures, achieving 0.75 accuracy and 0.74 AUC on I-SPY test data. N4 bias field correction slightly degraded performance. Without fine-tuning, external validation yielded 0.66 AUC, demonstrating cross-institutional generalizability. These findings highlight the effectiveness of dual-attention mechanisms in capturing transferable spatiotemporal features for HER2 stratification, advancing reproducible deep learning biomarkers in breast cancer imaging. 

**Abstract (ZH)**: HER2状态从动态对比增强MRI的无创预测：三头双注意力ResNet在乳腺癌影像中的应用 

---
