# Learning a General Model: Folding Clothing with Topological Dynamics 

**Title (ZH)**: 学习通用模型：基于拓扑动力学的衣物折叠 

**Authors**: Yiming Liu, Lijun Han, Enlin Gu, Hesheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.20720)  

**Abstract**: The high degrees of freedom and complex structure of garments present significant challenges for clothing manipulation. In this paper, we propose a general topological dynamics model to fold complex clothing. By utilizing the visible folding structure as the topological skeleton, we design a novel topological graph to represent the clothing state. This topological graph is low-dimensional and applied for complex clothing in various folding states. It indicates the constraints of clothing and enables predictions regarding clothing movement. To extract graphs from self-occlusion, we apply semantic segmentation to analyze the occlusion relationships and decompose the clothing structure. The decomposed structure is then combined with keypoint detection to generate the topological graph. To analyze the behavior of the topological graph, we employ an improved Graph Neural Network (GNN) to learn the general dynamics. The GNN model can predict the deformation of clothing and is employed to calculate the deformation Jacobi matrix for control. Experiments using jackets validate the algorithm's effectiveness to recognize and fold complex clothing with self-occlusion. 

**Abstract (ZH)**: 高自由度和复杂结构的服装对人体形操作提出显著挑战。本文提出了一种通用拓扑动力学模型以折叠复杂服装。通过利用可见折叠结构作为拓扑骨架，我们设计了一种新型拓扑图来表示服装状态。该拓扑图低维度且适用于各种折叠状态的复杂服装，可以表明服装的约束并使服装运动预测成为可能。为了从自遮挡中提取图形，我们应用语义分割来分析遮挡关系并分解服装结构。分解后的结构随后与关键点检测结合以生成拓扑图。为了分析拓扑图的行为，我们采用改进的图神经网络（GNN）来学习一般动力学。GNN模型可以预测服装变形并用于计算变形雅可比矩阵以进行控制。使用夹克进行的实验验证了该算法在识别和折叠具有自遮挡的复杂服装时的有效性。 

---
# PRISM-DP: Spatial Pose-based Observations for Diffusion-Policies via Segmentation, Mesh Generation, and Pose Tracking 

**Title (ZH)**: PRISM-DP: 基于空间姿态的观察方法以通过分割、网格生成和姿态跟踪实现扩散策略 

**Authors**: Xiatao Sun, Yinxing Chen, Daniel Rakita  

**Link**: [PDF](https://arxiv.org/pdf/2504.20359)  

**Abstract**: Diffusion-based visuomotor policies generate robot motions by learning to denoise action-space trajectories conditioned on observations. These observations are commonly streams of RGB images, whose high dimensionality includes substantial task-irrelevant information, requiring large models to extract relevant patterns. In contrast, using more structured observations, such as the spatial poses (positions and orientations) of key objects over time, enables training more compact policies that can recognize relevant patterns with fewer parameters. However, obtaining accurate object poses in open-set, real-world environments remains challenging. For instance, it is impractical to assume that all relevant objects are equipped with markers, and recent learning-based 6D pose estimation and tracking methods often depend on pre-scanned object meshes, requiring manual reconstruction. In this work, we propose PRISM-DP, an approach that leverages segmentation, mesh generation, pose estimation, and pose tracking models to enable compact diffusion policy learning directly from the spatial poses of task-relevant objects. Crucially, because PRISM-DP uses a mesh generation model, it eliminates the need for manual mesh processing or creation, improving scalability and usability in open-set, real-world environments. Experiments across a range of tasks in both simulation and real-world settings show that PRISM-DP outperforms high-dimensional image-based diffusion policies and achieves performance comparable to policies trained with ground-truth state information. We conclude with a discussion of the broader implications and limitations of our approach. 

**Abstract (ZH)**: 基于扩散的视听运动策略通过学习条件于观察的行动空间轨迹去噪来生成机器人运动。这些观察通常是RGB图像流，其高维度中包含大量与任务无关的信息，需要大型模型来提取相关模式。相比之下，使用更结构化的观察，如随时间变化的关键对象的空间姿态（位置和方向），能够训练更紧凑的策略，并用较少的参数识别相关模式。然而，在开放集的实际环境中获得准确的对象姿态仍然具有挑战性。例如，并非所有相关对象都配备标记，基于学习的6D姿态估计算法通常依赖于预扫描的对象网格，需要手动重建。在本工作中，我们提出了一种名为PRISM-DP的方法，该方法利用分割、网格生成、姿态估计和姿态跟踪模型，直接从任务相关对象的空间姿态中进行紧凑扩散策略学习。关键的是，由于PRISM-DP使用了网格生成模型，它消除了手动网格处理或创建的需求，提高了在开放集的实际环境中的可扩展性和易用性。在仿真和实际环境中的多种任务实验中，PRISM-DP优于高维度图像基扩散策略，并实现了与基于真实状态信息训练的策略相当的性能。本文最后讨论了我们方法的更广泛影响和局限性。 

---
# GSFeatLoc: Visual Localization Using Feature Correspondence on 3D Gaussian Splatting 

**Title (ZH)**: GSFeatLoc: 基于3D高斯绘制特征对应的空间定位 

**Authors**: Jongwon Lee, Timothy Bretl  

**Link**: [PDF](https://arxiv.org/pdf/2504.20379)  

**Abstract**: In this paper, we present a method for localizing a query image with respect to a precomputed 3D Gaussian Splatting (3DGS) scene representation. First, the method uses 3DGS to render a synthetic RGBD image at some initial pose estimate. Second, it establishes 2D-2D correspondences between the query image and this synthetic image. Third, it uses the depth map to lift the 2D-2D correspondences to 2D-3D correspondences and solves a perspective-n-point (PnP) problem to produce a final pose estimate. Results from evaluation across three existing datasets with 38 scenes and over 2,700 test images show that our method significantly reduces both inference time (by over two orders of magnitude, from more than 10 seconds to as fast as 0.1 seconds) and estimation error compared to baseline methods that use photometric loss minimization. Results also show that our method tolerates large errors in the initial pose estimate of up to 55° in rotation and 1.1 units in translation (normalized by scene scale), achieving final pose errors of less than 5° in rotation and 0.05 units in translation on 90% of images from the Synthetic NeRF and Mip-NeRF360 datasets and on 42% of images from the more challenging Tanks and Temples dataset. 

**Abstract (ZH)**: 本文提出了一种基于预先计算的3D高斯点云表示的查询图像定位方法。首先，该方法使用3D高斯点云渲染初始姿态估计下的合成RGBD图像。其次，它在查询图像与合成图像之间建立2D-2D对应关系。然后，利用深度图将2D-2D对应关系提升为2D-3D对应关系，并求解透视n点问题（PnP问题）以生成最终的姿态估计。在三个现有数据集上的评估结果显示，与使用 photometric损失最小化的基线方法相比，该方法显著减少了推理时间（减少了两个数量级以上，从超过10秒加速到0.1秒以内）并降低了估计误差。结果还显示，该方法可以容忍初始姿态估计的大误差，旋转误差高达55°，平移误差高达1.1个单位（以场景尺度归一化），在90%的Synthetic NeRF和Mip-NeRF360数据集图像以及42%的更具挑战性的Tanks and Temples数据集图像上，最终的姿态误差分别小于5°旋转和0.05个单位平移。 

---
# Improving trajectory continuity in drone-based crowd monitoring using a set of minimal-cost techniques and deep discriminative correlation filters 

**Title (ZH)**: 基于一组最小成本技术与深度区分性相关滤波器提高无人机人群监控轨迹连续性 

**Authors**: Bartosz Ptak, Marek Kraft  

**Link**: [PDF](https://arxiv.org/pdf/2504.20234)  

**Abstract**: Drone-based crowd monitoring is the key technology for applications in surveillance, public safety, and event management. However, maintaining tracking continuity and consistency remains a significant challenge. Traditional detection-assignment tracking methods struggle with false positives, false negatives, and frequent identity switches, leading to degraded counting accuracy and making in-depth analysis impossible. This paper introduces a point-oriented online tracking algorithm that improves trajectory continuity and counting reliability in drone-based crowd monitoring. Our method builds on the Simple Online and Real-time Tracking (SORT) framework, replacing the original bounding-box assignment with a point-distance metric. The algorithm is enhanced with three cost-effective techniques: camera motion compensation, altitude-aware assignment, and classification-based trajectory validation. Further, Deep Discriminative Correlation Filters (DDCF) that re-use spatial feature maps from localisation algorithms for increased computational efficiency through neural network resource sharing are integrated to refine object tracking by reducing noise and handling missed detections. The proposed method is evaluated on the DroneCrowd and newly shared UP-COUNT-TRACK datasets, demonstrating substantial improvements in tracking metrics, reducing counting errors to 23% and 15%, respectively. The results also indicate a significant reduction of identity switches while maintaining high tracking accuracy, outperforming baseline online trackers and even an offline greedy optimisation method. 

**Abstract (ZH)**: 基于无人机的人群监测中点导向的在线跟踪算法：提高轨迹连续性和计数可靠性 

---
# A Picture is Worth a Thousand Prompts? Efficacy of Iterative Human-Driven Prompt Refinement in Image Regeneration Tasks 

**Title (ZH)**: 一张图片抵得上一千个提示？图像再生任务中迭代的人工驱动提示 refinement 的有效性 

**Authors**: Khoi Trinh, Scott Seidenberger, Raveen Wijewickrama, Murtuza Jadliwala, Anindya Maiti  

**Link**: [PDF](https://arxiv.org/pdf/2504.20340)  

**Abstract**: With AI-generated content becoming ubiquitous across the web, social media, and other digital platforms, it is vital to examine how such content are inspired and generated. The creation of AI-generated images often involves refining the input prompt iteratively to achieve desired visual outcomes. This study focuses on the relatively underexplored concept of image regeneration using AI, in which a human operator attempts to closely recreate a specific target image by iteratively refining their prompt. Image regeneration is distinct from normal image generation, which lacks any predefined visual reference. A separate challenge lies in determining whether existing image similarity metrics (ISMs) can provide reliable, objective feedback in iterative workflows, given that we do not fully understand if subjective human judgments of similarity align with these metrics. Consequently, we must first validate their alignment with human perception before assessing their potential as a feedback mechanism in the iterative prompt refinement process. To address these research gaps, we present a structured user study evaluating how iterative prompt refinement affects the similarity of regenerated images relative to their targets, while also examining whether ISMs capture the same improvements perceived by human observers. Our findings suggest that incremental prompt adjustments substantially improve alignment, verified through both subjective evaluations and quantitative measures, underscoring the broader potential of iterative workflows to enhance generative AI content creation across various application domains. 

**Abstract (ZH)**: 随着生成式AI内容在互联网、社交媒体和其他数字平台上的普遍应用，亟需考察此类内容的灵感来源及其生成过程。本研究重点关注使用AI进行图像再生的概念，其中人类操作者通过迭代细化提示词试图精确还原特定目标图像。图像再生与缺乏预定义视觉参考的普通图像生成不同。另一个挑战在于，现有图像相似度度量(ISMs)是否能在迭代工作流程中提供可靠且客观的反馈，鉴于我们尚未完全理解主观的人类相似度判断是否与这些度量相一致。因此，我们首先需要验证它们是否与人类感知一致，然后再评估它们作为迭代提示词细化过程中的反馈机制的潜在应用。为填补这些研究空白，我们开展了一项结构化的用户研究，以评估迭代提示词细化如何影响再生图像与目标图像的相似度，并探讨ISMs是否能够捕捉到人类观察者感知到的相同改进。研究发现，逐步调整提示词显著提升了相似度的一致性，通过主观评价和定量指标均得到了验证，突显了迭代工作流程在各种应用领域增强生成式AI内容创作的广泛潜力。 

---
# Return Capping: Sample-Efficient CVaR Policy Gradient Optimisation 

**Title (ZH)**: 返回上限设置：基于样本高效的CVaR策略梯度优化 

**Authors**: Harry Mead, Clarissa Costen, Bruno Lacerda, Nick Hawes  

**Link**: [PDF](https://arxiv.org/pdf/2504.20887)  

**Abstract**: When optimising for conditional value at risk (CVaR) using policy gradients (PG), current meth- ods rely on discarding a large proportion of tra- jectories, resulting in poor sample efficiency. We propose a reformulation of the CVaR optimisation problem by capping the total return of trajecto- ries used in training, rather than simply discard- ing them, and show that this is equivalent to the original problem if the cap is set appropriately. We show, with empirical results in an number of environments, that this reformulation of the prob- lem results in consistently improved performance compared to baselines. 

**Abstract (ZH)**: 通过策略梯度优化条件价值 at 风险（CVaR）的问题重述：提高样本效率的方法 

---
# RadSAM: Segmenting 3D radiological images with a 2D promptable model 

**Title (ZH)**: RadSAM: 使用可提示的2D模型分割3D放射影像 

**Authors**: Julien Khlaut, Elodie Ferreres, Daniel Tordjman, Hélène Philippe, Tom Boeken, Pierre Manceron, Corentin Dancette  

**Link**: [PDF](https://arxiv.org/pdf/2504.20837)  

**Abstract**: Medical image segmentation is a crucial and time-consuming task in clinical care, where mask precision is extremely important. The Segment Anything Model (SAM) offers a promising approach, as it provides an interactive interface based on visual prompting and edition to refine an initial segmentation. This model has strong generalization capabilities, does not rely on predefined classes, and adapts to diverse objects; however, it is pre-trained on natural images and lacks the ability to process medical data effectively. In addition, this model is built for 2D images, whereas a whole medical domain is based on 3D images, such as CT and MRI. Recent adaptations of SAM for medical imaging are based on 2D models, thus requiring one prompt per slice to segment 3D objects, making the segmentation process tedious. They also lack important features such as editing. To bridge this gap, we propose RadSAM, a novel method for segmenting 3D objects with a 2D model from a single prompt. In practice, we train a 2D model using noisy masks as initial prompts, in addition to bounding boxes and points. We then use this novel prompt type with an iterative inference pipeline to reconstruct the 3D mask slice-by-slice. We introduce a benchmark to evaluate the model's ability to segment 3D objects in CT images from a single prompt and evaluate the models' out-of-domain transfer and edition capabilities. We demonstrate the effectiveness of our approach against state-of-the-art models on this benchmark using the AMOS abdominal organ segmentation dataset. 

**Abstract (ZH)**: 一种基于单个提示的2D模型分割3D医疗对象的方法：RadSAM 

---
# Advance Fake Video Detection via Vision Transformers 

**Title (ZH)**: 基于视觉变换器的先进虚假视频检测 

**Authors**: Joy Battocchio, Stefano Dell'Anna, Andrea Montibeller, Giulia Boato  

**Link**: [PDF](https://arxiv.org/pdf/2504.20669)  

**Abstract**: Recent advancements in AI-based multimedia generation have enabled the creation of hyper-realistic images and videos, raising concerns about their potential use in spreading misinformation. The widespread accessibility of generative techniques, which allow for the production of fake multimedia from prompts or existing media, along with their continuous refinement, underscores the urgent need for highly accurate and generalizable AI-generated media detection methods, underlined also by new regulations like the European Digital AI Act. In this paper, we draw inspiration from Vision Transformer (ViT)-based fake image detection and extend this idea to video. We propose an {original} %innovative framework that effectively integrates ViT embeddings over time to enhance detection performance. Our method shows promising accuracy, generalization, and few-shot learning capabilities across a new, large and diverse dataset of videos generated using five open source generative techniques from the state-of-the-art, as well as a separate dataset containing videos produced by proprietary generative methods. 

**Abstract (ZH)**: 基于AI的多媒体生成Recent进展引发了对其潜在滥用的担忧，尤其是在传播虚假信息方面的应用。生成技术的广泛应用及其不断改进促使我们需要开发高准确性和普适性的AI生成媒体检测方法，这一需求也被新的法规如欧洲数字AI法案所强调。在本文中，我们受Vision Transformer (ViT) 基础的虚假图像检测启发，将其理念拓展至视频领域。我们提出了一种创新框架，有效结合了时间上的ViT嵌入，以提高检测性能。该方法在使用五种开源生成技术生成的大型多样数据集以及包含由专有生成方法制作的视频的数据集中都表现出良好的准确率、泛化能力和少量样本学习能力。 

---
# SCOPE-MRI: Bankart Lesion Detection as a Case Study in Data Curation and Deep Learning for Challenging Diagnoses 

**Title (ZH)**: SCOPE-MRI: 韦氏脱位检测及其在具有挑战性的诊断数据整理和深度学习研究中的案例分析 

**Authors**: Sahil Sethi, Sai Reddy, Mansi Sakarvadia, Jordan Serotte, Darlington Nwaudo, Nicholas Maassen, Lewis Shi  

**Link**: [PDF](https://arxiv.org/pdf/2504.20405)  

**Abstract**: While deep learning has shown strong performance in musculoskeletal imaging, existing work has largely focused on pathologies where diagnosis is not a clinical challenge, leaving more difficult problems underexplored, such as detecting Bankart lesions (anterior-inferior glenoid labral tears) on standard MRIs. Diagnosing these lesions is challenging due to their subtle imaging features, often leading to reliance on invasive MRI arthrograms (MRAs). This study introduces ScopeMRI, the first publicly available, expert-annotated dataset for shoulder pathologies, and presents a deep learning (DL) framework for detecting Bankart lesions on both standard MRIs and MRAs. ScopeMRI includes 586 shoulder MRIs (335 standard, 251 MRAs) from 558 patients who underwent arthroscopy. Ground truth labels were derived from intraoperative findings, the gold standard for diagnosis. Separate DL models for MRAs and standard MRIs were trained using a combination of CNNs and transformers. Predictions from sagittal, axial, and coronal views were ensembled to optimize performance. The models were evaluated on a 20% hold-out test set (117 MRIs: 46 MRAs, 71 standard MRIs). The models achieved an AUC of 0.91 and 0.93, sensitivity of 83% and 94%, and specificity of 91% and 86% for standard MRIs and MRAs, respectively. Notably, model performance on non-invasive standard MRIs matched or surpassed radiologists interpreting MRAs. External validation demonstrated initial generalizability across imaging protocols. This study demonstrates that DL models can achieve radiologist-level diagnostic performance on standard MRIs, reducing the need for invasive MRAs. By releasing ScopeMRI and a modular codebase for training and evaluating deep learning models on 3D medical imaging data, we aim to accelerate research in musculoskeletal imaging and support the development of new datasets for clinically challenging diagnostic tasks. 

**Abstract (ZH)**: 基于深学习检测标准MRI和MRAr中的Bankart损伤：ScopeMRI数据集与研究 

---
# Decoding Latent Spaces: Assessing the Interpretability of Time Series Foundation Models for Visual Analytics 

**Title (ZH)**: 解码潜在空间：时间序列基础模型在可视化分析中的可解释性评估 

**Authors**: Inmaculada Santamaria-Valenzuela, Victor Rodriguez-Fernandez, Javier Huertas-Tato, Jong Hyuk Park, David Camacho  

**Link**: [PDF](https://arxiv.org/pdf/2504.20099)  

**Abstract**: The present study explores the interpretability of latent spaces produced by time series foundation models, focusing on their potential for visual analysis tasks. Specifically, we evaluate the MOMENT family of models, a set of transformer-based, pre-trained architectures for multivariate time series tasks such as: imputation, prediction, classification, and anomaly detection. We evaluate the capacity of these models on five datasets to capture the underlying structures in time series data within their latent space projection and validate whether fine tuning improves the clarity of the resulting embedding spaces. Notable performance improvements in terms of loss reduction were observed after fine tuning. Visual analysis shows limited improvement in the interpretability of the embeddings, requiring further work. Results suggest that, although Time Series Foundation Models such as MOMENT are robust, their latent spaces may require additional methodological refinements to be adequately interpreted, such as alternative projection techniques, loss functions, or data preprocessing strategies. Despite the limitations of MOMENT, foundation models supose a big reduction in execution time and so a great advance for interactive visual analytics. 

**Abstract (ZH)**: 本研究探讨了时间序列基础模型生成的潜在空间的可解释性，着重于其在视觉分析任务中的应用潜力。具体而言，我们评估了MOMENT模型家族，这是一种基于变换器的预训练架构，适用于多变量时间序列任务，如插值、预测、分类和异常检测。我们在这五个数据集中评估了这些模型在潜在空间投影中捕获时间序列数据的潜在结构的能力，并验证了微调是否能改善最终嵌入空间的清晰度。微调后观察到损失减小的显著性能提升。视觉分析显示嵌入的可解释性改进有限，需要进一步研究。结果表明，尽管如MOMENT的时间序列基础模型具有鲁棒性，但其潜在空间可能需要额外的方法学改进以充分解释，例如替代投影技术、损失函数或数据预处理策略。尽管MOMENT存在局限性，但基础模型仍假定了执行时间的大规模减少，这是交互式视觉分析的一大进步。 

---
