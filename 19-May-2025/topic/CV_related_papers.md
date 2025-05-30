# Exploiting Radiance Fields for Grasp Generation on Novel Synthetic Views 

**Title (ZH)**: 利用辐射场生成新型合成视图上的 grasping 操作 

**Authors**: Abhishek Kashyap, Henrik Andreasson, Todor Stoyanov  

**Link**: [PDF](https://arxiv.org/pdf/2505.11467)  

**Abstract**: Vision based robot manipulation uses cameras to capture one or more images of a scene containing the objects to be manipulated. Taking multiple images can help if any object is occluded from one viewpoint but more visible from another viewpoint. However, the camera has to be moved to a sequence of suitable positions for capturing multiple images, which requires time and may not always be possible, due to reachability constraints. So while additional images can produce more accurate grasp poses due to the extra information available, the time-cost goes up with the number of additional views sampled. Scene representations like Gaussian Splatting are capable of rendering accurate photorealistic virtual images from user-specified novel viewpoints. In this work, we show initial results which indicate that novel view synthesis can provide additional context in generating grasp poses. Our experiments on the Graspnet-1billion dataset show that novel views contributed force-closure grasps in addition to the force-closure grasps obtained from sparsely sampled real views while also improving grasp coverage. In the future we hope this work can be extended to improve grasp extraction from radiance fields constructed with a single input image, using for example diffusion models or generalizable radiance fields. 

**Abstract (ZH)**: 基于视觉的机器人操作通过相机捕捉包含待操作物体的一个或多个场景图像。多张图像的获取有助于遮挡物体在某视角不可见而在另一视角可见的情况，但相机需要移动到一系列适合的位置来捕捉多张图像，这需要时间，且由于可达性约束，不一定总能实现。因此，虽然额外的图像可以提供更多的信息从而更准确地生成抓取姿态，但额外视角的数量也增加了时间成本。场景表示法如Gaussian Splatting能够从用户指定的新视角渲染出准确的逼真虚拟图像。在本文中，我们展示了初步结果，表明新的视角合成能够为生成抓取姿态提供额外的上下文信息。我们的实验在Graspnet-1billion数据集上表明，新视角不仅提供了从稀疏采样的真实视角获得的力闭合抓取，还提高了抓取覆盖率。未来，我们希望这项工作能够扩展应用，通过例如扩散模型或通用辐射场从单张输入图像构建的辐射场中提取抓取。 

---
# SurgPose: Generalisable Surgical Instrument Pose Estimation using Zero-Shot Learning and Stereo Vision 

**Title (ZH)**: SurgPose: 任意场景下基于零样本学习和立体视觉的手术器械姿态估计 

**Authors**: Utsav Rai, Haozheng Xu, Stamatia Giannarou  

**Link**: [PDF](https://arxiv.org/pdf/2505.11439)  

**Abstract**: Accurate pose estimation of surgical tools in Robot-assisted Minimally Invasive Surgery (RMIS) is essential for surgical navigation and robot control. While traditional marker-based methods offer accuracy, they face challenges with occlusions, reflections, and tool-specific designs. Similarly, supervised learning methods require extensive training on annotated datasets, limiting their adaptability to new tools. Despite their success in other domains, zero-shot pose estimation models remain unexplored in RMIS for pose estimation of surgical instruments, creating a gap in generalising to unseen surgical tools. This paper presents a novel 6 Degrees of Freedom (DoF) pose estimation pipeline for surgical instruments, leveraging state-of-the-art zero-shot RGB-D models like the FoundationPose and SAM-6D. We advanced these models by incorporating vision-based depth estimation using the RAFT-Stereo method, for robust depth estimation in reflective and textureless environments. Additionally, we enhanced SAM-6D by replacing its instance segmentation module, Segment Anything Model (SAM), with a fine-tuned Mask R-CNN, significantly boosting segmentation accuracy in occluded and complex conditions. Extensive validation reveals that our enhanced SAM-6D surpasses FoundationPose in zero-shot pose estimation of unseen surgical instruments, setting a new benchmark for zero-shot RGB-D pose estimation in RMIS. This work enhances the generalisability of pose estimation for unseen objects and pioneers the application of RGB-D zero-shot methods in RMIS. 

**Abstract (ZH)**: 基于零样本RGB-D模型的手术工具六自由度姿态估计方法在机器人辅助微创手术中的应用 

---
# TACO: Rethinking Semantic Communications with Task Adaptation and Context Embedding 

**Title (ZH)**: TACO: 任务适应与上下文嵌入的语义通信 rethink 

**Authors**: Achintha Wijesinghe, Weiwei Wang, Suchinthaka Wanninayaka, Songyang Zhang, Zhi Ding  

**Link**: [PDF](https://arxiv.org/pdf/2505.10834)  

**Abstract**: Recent advancements in generative artificial intelligence have introduced groundbreaking approaches to innovating next-generation semantic communication, which prioritizes conveying the meaning of a message rather than merely transmitting raw data. A fundamental challenge in semantic communication lies in accurately identifying and extracting the most critical semantic information while adapting to downstream tasks without degrading performance, particularly when the objective at the receiver may evolve over time. To enable flexible adaptation to multiple tasks at the receiver, this work introduces a novel semantic communication framework, which is capable of jointly capturing task-specific information to enhance downstream task performance and contextual information. Through rigorous experiments on popular image datasets and computer vision tasks, our framework shows promising improvement compared to existing work, including superior performance in downstream tasks, better generalizability, ultra-high bandwidth efficiency, and low reconstruction latency. 

**Abstract (ZH)**: 近期生成式人工智能的发展引入了下一代语义通信的开创性方法，重点在于传达信息的意义而非仅传输原始数据。语义通信的基本挑战在于在适应下游任务时准确识别和提取最关键的信息，尤其是在接收方的目标可能随时间变化时，不降低性能。为了使接收方能够灵活适应多种任务，本文提出了一种新颖的语义通信框架，该框架能够联合捕捉任务特定信息以提升下游任务性能和上下文信息。通过在流行图像数据集和计算机视觉任务上的严格实验，我们的框架展现出了与现有工作相比的显著改进，包括在下游任务中的卓越性能、更好的泛化能力、超高的带宽效率和低重构延迟。 

---
# Improving Object Detection Performance through YOLOv8: A Comprehensive Training and Evaluation Study 

**Title (ZH)**: 通过YOLOv8提高目标检测性能：一项全面的训练与评估研究 

**Authors**: Rana Poureskandar, Shiva Razzagzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2505.11424)  

**Abstract**: This study evaluated the performance of a YOLOv8-based segmentation model for detecting and segmenting wrinkles in facial images. 

**Abstract (ZH)**: 本研究评估了基于YOLOv8的分割模型在检测和分割面部图像中皱纹的表现。 

---
# FALCON: False-Negative Aware Learning of Contrastive Negatives in Vision-Language Pretraining 

**Title (ZH)**: FALCON：在视觉-语言预训练中关注负样本的虚假阴性学习 

**Authors**: Myunsoo Kim, Seong-Woong Shim, Byung-Jun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.11192)  

**Abstract**: False negatives pose a critical challenge in vision-language pretraining (VLP) due to the many-to-many correspondence between images and texts in large-scale datasets. These false negatives introduce conflicting supervision signals that degrade the learned embedding space and diminish the effectiveness of hard negative sampling. In this paper, we propose FALCON (False-negative Aware Learning of COntrastive Negatives), a learning-based mini-batch construction strategy that adaptively balances the trade-off between hard and false negatives during VLP. Rather than relying on fixed heuristics, FALCON employs a negative mining scheduler that dynamically selects negative samples of appropriate hardness for each anchor instance during mini-batch construction, guided by a proxy for cross-modal alignment improvement. Experimental results demonstrate that FALCON significantly improves performance across two widely adopted VLP frameworks (ALBEF, BLIP-2) and a broad range of downstream tasks and evaluation settings, underscoring its effectiveness and robustness in mitigating the impact of false negatives. 

**Abstract (ZH)**: False Negatives在视觉-语言预训练中的关键挑战：FALCON（基于学习的False-negativeaware对比负样本学习）策略 

---
# CompAlign: Improving Compositional Text-to-Image Generation with a Complex Benchmark and Fine-Grained Feedback 

**Title (ZH)**: CompAlign: 通过复杂基准和细粒度反馈提高组合文本到图像生成性能 

**Authors**: Yixin Wan, Kai-Wei Chang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11178)  

**Abstract**: State-of-the-art T2I models are capable of generating high-resolution images given textual prompts. However, they still struggle with accurately depicting compositional scenes that specify multiple objects, attributes, and spatial relations. We present CompAlign, a challenging benchmark with an emphasis on assessing the depiction of 3D-spatial relationships, for evaluating and improving models on compositional image generation. CompAlign consists of 900 complex multi-subject image generation prompts that combine numerical and 3D-spatial relationships with varied attribute bindings. Our benchmark is remarkably challenging, incorporating generation tasks with 3+ generation subjects with complex 3D-spatial relationships. Additionally, we propose CompQuest, an interpretable and accurate evaluation framework that decomposes complex prompts into atomic sub-questions, then utilizes a MLLM to provide fine-grained binary feedback on the correctness of each aspect of generation elements in model-generated images. This enables precise quantification of alignment between generated images and compositional prompts. Furthermore, we propose an alignment framework that uses CompQuest's feedback as preference signals to improve diffusion models' compositional image generation abilities. Using adjustable per-image preferences, our method is easily scalable and flexible for different tasks. Evaluation of 9 T2I models reveals that: (1) models remarkable struggle more with compositional tasks with more complex 3D-spatial configurations, and (2) a noticeable performance gap exists between open-source accessible models and closed-source commercial models. Further empirical study on using CompAlign for model alignment yield promising results: post-alignment diffusion models achieve remarkable improvements in compositional accuracy, especially on complex generation tasks, outperforming previous approaches. 

**Abstract (ZH)**: 最先进的文本到图像（T2I）模型能够在给定文本提示的情况下生成高分辨率图像。然而，它们在准确描绘包含多个对象、属性和空间关系的组合场景方面仍然存在挑战。我们提出了CompAlign，一个专注于评估3D空间关系表示能力的具有挑战性的基准，用于评估和提升组合图像生成模型。CompAlign 包含900个复杂的多主体图像生成提示，结合了数值和3D空间关系，以及多样的属性绑定。我们的基准具有显著的挑战性，包括生成涉及3个及以上生成主体且具有复杂3D空间关系的任务。此外，我们提出了CompQuest，这是一种可解释且准确的评估框架，将复杂提示分解为原子子问题，然后利用多模态预训练语言模型（MLLM）提供生成元素在模型生成图像中的各个方面的细粒度二元反馈。这使得对生成图像与组合提示之间对齐的精确量化成为可能。此外，我们提出了一种利用CompQuest反馈作为偏好信号以改进扩散模型组合图像生成能力的框架。通过可调节的单个图像偏好，该方法易于扩展和适应不同的任务。评估9个T2I模型发现：(1) 模型在涉及更复杂3D空间配置的组合任务中表现尤为困难，(2) 开源可访问模型与封闭源商业模型之间存在明显的性能差距。进一步使用CompAlign进行模型对齐的经验研究表明，对齐后的扩散模型在组合准确性方面取得了显著改进，尤其是在复杂生成任务中超过了先前的方法。 

---
# CheX-DS: Improving Chest X-ray Image Classification with Ensemble Learning Based on DenseNet and Swin Transformer 

**Title (ZH)**: CheX-DS：基于DenseNet和Swin Transformer的集成学习改进胸部X光图像分类 

**Authors**: Xinran Li, Yu Liu, Xiujuan Xu, Xiaowei Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.11168)  

**Abstract**: The automatic diagnosis of chest diseases is a popular and challenging task. Most current methods are based on convolutional neural networks (CNNs), which focus on local features while neglecting global features. Recently, self-attention mechanisms have been introduced into the field of computer vision, demonstrating superior performance. Therefore, this paper proposes an effective model, CheX-DS, for classifying long-tail multi-label data in the medical field of chest X-rays. The model is based on the excellent CNN model DenseNet for medical imaging and the newly popular Swin Transformer model, utilizing ensemble deep learning techniques to combine the two models and leverage the advantages of both CNNs and Transformers. The loss function of CheX-DS combines weighted binary cross-entropy loss with asymmetric loss, effectively addressing the issue of data imbalance. The NIH ChestX-ray14 dataset is selected to evaluate the model's effectiveness. The model outperforms previous studies with an excellent average AUC score of 83.76\%, demonstrating its superior performance. 

**Abstract (ZH)**: 胸部疾病自动诊断是-popular-and-challenging-task-的一项流行且具有挑战性的任务。大多数现有方法基于卷积神经网络(CNNs)，侧重于局部特征而忽视了全局特征。最近，自注意力机制被引入计算机视觉领域，展现了卓越的性能。因此，本文提出了一种有效的模型CheX-DS，用于分类胸部X光片医学领域中长尾多标签数据。该模型基于用于医学成像的优秀CNN模型DenseNet和新兴流行的Swin Transformer模型，利用集成深度学习技术结合这两种模型，充分发挥CNN和Transformer的优势。CheX-DS的损失函数结合了加权二元交叉熵损失和非对称损失，有效解决了数据不平衡问题。该模型在NIH ChestX-ray14数据集上的评估结果显示，其平均AUC分数为83.76%，展示了其优越性能。 

---
# Attention on the Sphere 

**Title (ZH)**: 球面上的注意力 

**Authors**: Boris Bonev, Max Rietmann, Andrea Paris, Alberto Carpentieri, Thorsten Kurth  

**Link**: [PDF](https://arxiv.org/pdf/2505.11157)  

**Abstract**: We introduce a generalized attention mechanism for spherical domains, enabling Transformer architectures to natively process data defined on the two-dimensional sphere - a critical need in fields such as atmospheric physics, cosmology, and robotics, where preserving spherical symmetries and topology is essential for physical accuracy. By integrating numerical quadrature weights into the attention mechanism, we obtain a geometrically faithful spherical attention that is approximately rotationally equivariant, providing strong inductive biases and leading to better performance than Cartesian approaches. To further enhance both scalability and model performance, we propose neighborhood attention on the sphere, which confines interactions to geodesic neighborhoods. This approach reduces computational complexity and introduces the additional inductive bias for locality, while retaining the symmetry properties of our method. We provide optimized CUDA kernels and memory-efficient implementations to ensure practical applicability. The method is validated on three diverse tasks: simulating shallow water equations on the rotating sphere, spherical image segmentation, and spherical depth estimation. Across all tasks, our spherical Transformers consistently outperform their planar counterparts, highlighting the advantage of geometric priors for learning on spherical domains. 

**Abstract (ZH)**: 一种适用于球面域的广义注意力机制：增强变换器架构处理球面数据的能力 

---
# PhiNet v2: A Mask-Free Brain-Inspired Vision Foundation Model from Video 

**Title (ZH)**: PhiNet v2: 一种来自视频的无掩码脑启发视觉基础模型 

**Authors**: Makoto Yamada, Kian Ming A. Chai, Ayoub Rhim, Satoki Ishikawa, Mohammad Sabokrou, Yao-Hung Hubert Tsai  

**Link**: [PDF](https://arxiv.org/pdf/2505.11129)  

**Abstract**: Recent advances in self-supervised learning (SSL) have revolutionized computer vision through innovative architectures and learning objectives, yet they have not fully leveraged insights from biological visual processing systems. Recently, a brain-inspired SSL model named PhiNet was proposed; it is based on a ResNet backbone and operates on static image inputs with strong augmentation. In this paper, we introduce PhiNet v2, a novel Transformer-based architecture that processes temporal visual input (that is, sequences of images) without relying on strong augmentation. Our model leverages variational inference to learn robust visual representations from continuous input streams, similar to human visual processing. Through extensive experimentation, we demonstrate that PhiNet v2 achieves competitive performance compared to state-of-the-art vision foundation models, while maintaining the ability to learn from sequential input without strong data augmentation. This work represents a significant step toward more biologically plausible computer vision systems that process visual information in a manner more closely aligned with human cognitive processes. 

**Abstract (ZH)**: 最近在自监督学习（SSL）领域的进展通过创新的架构和学习目标革新了计算机视觉，但尚未充分利用生物视觉处理系统的见解。最近，一种受脑启发的SSL模型PhiNet被提出；它基于ResNet骨干网络，并对静态图像输入进行强增强操作。在本文中，我们引入了PhiNet v2，这是一种新型的基于Transformer的架构，能够处理时序视觉输入（即图像序列）而不依赖于强增强。我们的模型利用变分推断从连续输入流中学习鲁棒的视觉表示，类似于人类视觉处理。通过大量的实验，我们证明了PhiNet v2在与最先进的视觉基础模型相当的性能上，同时保持了从序列输入中学习的能力，而不需要强烈的数据增强。这项工作代表了朝着更符合生物学原理的计算机视觉系统迈进的重要一步，这些系统能够以更接近人类认知过程的方式处理视觉信息。 

---
# Towards Self-Improvement of Diffusion Models via Group Preference Optimization 

**Title (ZH)**: 通过群体偏好优化实现扩散模型的自我改进 

**Authors**: Renjie Chen, Wenfeng Lin, Yichen Zhang, Jiangchuan Wei, Boyuan Liu, Chao Feng, Jiao Ran, Mingyu Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.11070)  

**Abstract**: Aligning text-to-image (T2I) diffusion models with Direct Preference Optimization (DPO) has shown notable improvements in generation quality. However, applying DPO to T2I faces two challenges: the sensitivity of DPO to preference pairs and the labor-intensive process of collecting and annotating high-quality data. In this work, we demonstrate that preference pairs with marginal differences can degrade DPO performance. Since DPO relies exclusively on relative ranking while disregarding the absolute difference of pairs, it may misclassify losing samples as wins, or vice versa. We empirically show that extending the DPO from pairwise to groupwise and incorporating reward standardization for reweighting leads to performance gains without explicit data selection. Furthermore, we propose Group Preference Optimization (GPO), an effective self-improvement method that enhances performance by leveraging the model's own capabilities without requiring external data. Extensive experiments demonstrate that GPO is effective across various diffusion models and tasks. Specifically, combining with widely used computer vision models, such as YOLO and OCR, the GPO improves the accurate counting and text rendering capabilities of the Stable Diffusion 3.5 Medium by 20 percentage points. Notably, as a plug-and-play method, no extra overhead is introduced during inference. 

**Abstract (ZH)**: 使用Group Preference Optimization (GPO)提升文本到图像生成模型性能：无需数据选择的自我优化方法 

---
# Completely Weakly Supervised Class-Incremental Learning for Semantic Segmentation 

**Title (ZH)**: 完全弱监督类增量学习用于语义分割 

**Authors**: David Minkwan Kim, Soeun Lee, Byeongkeun Kang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10781)  

**Abstract**: This work addresses the task of completely weakly supervised class-incremental learning for semantic segmentation to learn segmentation for both base and additional novel classes using only image-level labels. While class-incremental semantic segmentation (CISS) is crucial for handling diverse and newly emerging objects in the real world, traditional CISS methods require expensive pixel-level annotations for training. To overcome this limitation, partially weakly-supervised approaches have recently been proposed. However, to the best of our knowledge, this is the first work to introduce a completely weakly-supervised method for CISS. To achieve this, we propose to generate robust pseudo-labels by combining pseudo-labels from a localizer and a sequence of foundation models based on their uncertainty. Moreover, to mitigate catastrophic forgetting, we introduce an exemplar-guided data augmentation method that generates diverse images containing both previous and novel classes with guidance. Finally, we conduct experiments in three common experimental settings: 15-5 VOC, 10-10 VOC, and COCO-to-VOC, and in two scenarios: disjoint and overlap. The experimental results demonstrate that our completely weakly supervised method outperforms even partially weakly supervised methods in the 15-5 VOC and 10-10 VOC settings while achieving competitive accuracy in the COCO-to-VOC setting. 

**Abstract (ZH)**: 本研究解决了完全弱监督类增量学习在语义分割中的问题，仅使用图层面标签来学习基类和新增类的分割。虽然类增量语义分割（CISS）对于处理现实世界中多种新兴对象至关重要，但传统CISS方法需要昂贵的像素级标注进行训练。为了克服这一局限性，最近提出了部分弱监督方法。然而，据我们所知，这是首次提出完全弱监督方法用于CISS。为了实现这一点，我们提出了一种结合局部化器和一系列基于不确定性基础模型的伪标签生成稳健伪标签的方法。此外，为缓解灾难性遗忘，我们引入了一种基于示例的数据增强方法，该方法在指导下生成包含以前和新增类的多样化图像。最后，我们在三个常见实验设置（15-5 VOC、10-10 VOC和COCO-to-VOC）和两种场景（分离和重叠）下进行了实验。实验结果表明，在15-5 VOC和10-10 VOC设置中，我们的完全弱监督方法在性能上优于部分弱监督方法，在COCO-to-VOC设置中，我们的方法达到了可竞争的准确率。 

---
# CLIP Embeddings for AI-Generated Image Detection: A Few-Shot Study with Lightweight Classifier 

**Title (ZH)**: 基于CLIP嵌入的AI生成图像检测：一种轻量级分类器参与的少样本研究 

**Authors**: Ziyang Ou  

**Link**: [PDF](https://arxiv.org/pdf/2505.10664)  

**Abstract**: Verifying the authenticity of AI-generated images presents a growing challenge on social media platforms these days. While vision-language models (VLMs) like CLIP outdo in multimodal representation, their capacity for AI-generated image classification is underexplored due to the absence of such labels during the pre-training process. This work investigates whether CLIP embeddings inherently contain information indicative of AI generation. A proposed pipeline extracts visual embeddings using a frozen CLIP model, feeds its embeddings to lightweight networks, and fine-tunes only the final classifier. Experiments on the public CIFAKE benchmark show the performance reaches 95% accuracy without language reasoning. Few-shot adaptation to curated custom with 20% of the data results in performance to 85%. A closed-source baseline (Gemini-2.0) has the best zero-shot accuracy yet fails on specific styles. Notably, some specific image types, such as wide-angle photographs and oil paintings, pose significant challenges to classification. These results indicate previously unexplored difficulties in classifying certain types of AI-generated images, revealing new and more specific questions in this domain that are worth further investigation. 

**Abstract (ZH)**: 验证AI生成图像的真实性目前在社交媒体平台上越来越成为一个挑战。尽管像CLIP这样的多模态模型在多模态表示方面表现出色，但由于预训练过程中缺乏相应的标签，它们在AI生成图像分类方面的能力尚未得到充分利用。本文探讨了CLIP嵌入是否包含指示AI生成的信息。提出了一种pipeline流程，使用冻结的CLIP模型提取视觉嵌入，将其嵌入传递给轻量级网络，并仅 fine-tune 最终分类器。在公共CIFAKE基准测试上的实验显示，无需语言推理即可达到95%的准确性。利用20%的定制数据进行少样本适应，性能达到85%。公开源码基线（Gemini-2.0）在零样本准确性上表现最佳，但在特定风格上失败。值得注意的是，某些特定图像类型，如广角照片和油画，给分类带来了重大挑战。这些结果表明，在分类某些类型的AI生成图像时存在尚未探索的困难，揭示了值得进一步调查的新且更具体的领域问题。 

---
# Super-Resolution Generative Adversarial Networks based Video Enhancement 

**Title (ZH)**: 基于超分辨率生成对抗网络的视频增强 

**Authors**: Kağan ÇETİN  

**Link**: [PDF](https://arxiv.org/pdf/2505.10589)  

**Abstract**: This study introduces an enhanced approach to video super-resolution by extending ordinary Single-Image Super-Resolution (SISR) Super-Resolution Generative Adversarial Network (SRGAN) structure to handle spatio-temporal data. While SRGAN has proven effective for single-image enhancement, its design does not account for the temporal continuity required in video processing. To address this, a modified framework that incorporates 3D Non-Local Blocks is proposed, which is enabling the model to capture relationships across both spatial and temporal dimensions. An experimental training pipeline is developed, based on patch-wise learning and advanced data degradation techniques, to simulate real-world video conditions and learn from both local and global structures and details. This helps the model generalize better and maintain stability across varying video content while maintaining the general structure besides the pixel-wise correctness. Two model variants-one larger and one more lightweight-are presented to explore the trade-offs between performance and efficiency. The results demonstrate improved temporal coherence, sharper textures, and fewer visual artifacts compared to traditional single-image methods. This work contributes to the development of practical, learning-based solutions for video enhancement tasks, with potential applications in streaming, gaming, and digital restoration. 

**Abstract (ZH)**: 该研究提出了一种增强的视频超分辨率方法，通过将普通的单图像超分辨率（SISR）生成对抗网络（SRGAN）结构扩展为处理时空数据的框架。虽然SRGAN在单图像增强方面已 proven 有效，但其设计未考虑视频处理所需的时序连续性。为此，提出了一种结合3D非局部块的改进框架，使模型能够在时空维度上捕捉关系。基于块学习和高级数据降质技术开发了实验训练管道，以模拟真实世界的视频条件，并从局部和全局结构与细节中学习，从而帮助模型更好地泛化并在不同视频内容下保持稳定，同时保持像素级的准确性。提出了两种模型变体——一个更大且一个更轻量级——以探索性能与效率之间的权衡。结果表明，与传统单图像方法相比，这种方法在时间一致性、锐利纹理和较少的视觉伪影方面表现出改进。该工作为视频增强任务提供了实用的基于学习的解决方案，具有潜在的应用价值，如流媒体、游戏和数字修复等领域。 

---
# GRNN:Recurrent Neural Network based on Ghost Features for Video Super-Resolution 

**Title (ZH)**: 基于幽灵特征的递归神经网络Video Super-分辨率 

**Authors**: Yutong Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.10577)  

**Abstract**: Modern video super-resolution (VSR) systems based on convolutional neural networks (CNNs) require huge computational costs. The problem of feature redundancy is present in most models in many domains, but is rarely discussed in VSR. We experimentally observe that many features in VSR models are also similar to each other, so we propose to use "Ghost features" to reduce this redundancy. We also analyze the so-called "gradient disappearance" phenomenon generated by the conventional recurrent convolutional network (RNN) model, and combine the Ghost module with RNN to complete the modeling on time series. The current frame is used as input to the model together with the next frame, the output of the previous frame and the hidden state. Extensive experiments on several benchmark models and datasets show that the PSNR and SSIM of our proposed modality are improved to some extent. Some texture details in the video are also better preserved. 

**Abstract (ZH)**: 基于卷积神经网络的现代视频超分辨率（VSR）系统需要巨大的计算成本。在许多领域中，特征冗余问题普遍存在，但在VSR中却鲜少讨论。我们实验观察到VSR模型中的许多特征彼此也很相似，因此我们提出使用“Ghost特征”来减少这种冗余。我们还分析了由传统递归卷积神经网络（RNN）模型产生的所谓的“梯度消失”现象，并将Ghost模块与RNN结合，用于时间序列建模。当前帧与下一帧一起作为模型输入，输出前一帧的输出和隐藏状态。在多个基准模型和数据集上的 extensive 实验显示，我们提出的方法在一定程度上提高了 PSNR 和 SSIM，并更好地保留了视频中的某些纹理细节。 

---
