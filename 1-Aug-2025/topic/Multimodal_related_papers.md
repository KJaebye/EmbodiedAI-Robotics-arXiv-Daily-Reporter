# User Experience Estimation in Human-Robot Interaction Via Multi-Instance Learning of Multimodal Social Signals 

**Title (ZH)**: 基于多实例学习的多模态社会信号在人机交互中的用户体验评估 

**Authors**: Ryo Miyoshi, Yuki Okafuji, Takuya Iwamoto, Junya Nakanishi, Jun Baba  

**Link**: [PDF](https://arxiv.org/pdf/2507.23544)  

**Abstract**: In recent years, the demand for social robots has grown, requiring them to adapt their behaviors based on users' states. Accurately assessing user experience (UX) in human-robot interaction (HRI) is crucial for achieving this adaptability. UX is a multi-faceted measure encompassing aspects such as sentiment and engagement, yet existing methods often focus on these individually. This study proposes a UX estimation method for HRI by leveraging multimodal social signals. We construct a UX dataset and develop a Transformer-based model that utilizes facial expressions and voice for estimation. Unlike conventional models that rely on momentary observations, our approach captures both short- and long-term interaction patterns using a multi-instance learning framework. This enables the model to capture temporal dynamics in UX, providing a more holistic representation. Experimental results demonstrate that our method outperforms third-party human evaluators in UX estimation. 

**Abstract (ZH)**: 近年来，对社会机器人的需求增长，要求它们根据用户状态调整行为。准确评估人类-机器人交互（HRI）中的用户体验（UX）对于实现这一适应性至关重要。UX是一项多方面的衡量标准，涵盖了情感和参与度等维度，但现有的方法往往侧重于分别评估这些方面。本研究提出了一种利用多模态社会信号的UX估计算法，构建了一个UX数据集，并开发了一个基于Transformer的模型，该模型利用面部表情和语音进行评估。与依赖于即时观察的传统模型不同，我们的方法使用多实例学习框架捕捉短期和长期的交互模式，从而能够捕捉UX的时间动态，提供更全面的表示。实验结果表明，我们的方法在UX评估中优于第三方人类评估者。 

---
# AGA: An adaptive group alignment framework for structured medical cross-modal representation learning 

**Title (ZH)**: AGA：一种自适应组对齐框架用于结构化医学跨模态表示学习 

**Authors**: Wei Li, Xun Gong, Jiao Li, Xiaobin Sun  

**Link**: [PDF](https://arxiv.org/pdf/2507.23402)  

**Abstract**: Learning medical visual representations from paired images and reports is a promising direction in representation learning. However, current vision-language pretraining methods in the medical domain often simplify clinical reports into single entities or fragmented tokens, ignoring their inherent structure. In addition, contrastive learning frameworks typically depend on large quantities of hard negative samples, which is impractical for small-scale medical datasets. To tackle these challenges, we propose Adaptive Grouped Alignment (AGA), a new framework that captures structured semantics from paired medical images and reports. AGA introduces a bidirectional grouping mechanism based on a sparse similarity matrix. For each image-report pair, we compute fine-grained similarities between text tokens and image patches. Each token selects its top-matching patches to form a visual group, and each patch selects its most related tokens to form a language group. To enable adaptive grouping, we design two threshold gating modules, called Language Grouped Threshold Gate and Vision Grouped Threshold Gate, which learn grouping thresholds dynamically. Group representations are computed as weighted averages based on similarity scores. To align each token with its group representation, we introduce an Instance Aware Group Alignment loss that operates within each image-text pair, removing the need for external negatives. Finally, a Bidirectional Cross-modal Grouped Alignment module is applied to enhance fine-grained alignment between visual and linguistic group representations. Extensive experiments on public and private datasets show that our method achieves strong performance on image-text retrieval and classification tasks under both fine-tuning and zero-shot settings. 

**Abstract (ZH)**: 基于配对图像和报告学习医学视觉表示是一种有前景的表示学习方向。然而，当前医学领域的视觉-语言预训练方法常常将临床报告简化为单个实体或碎片化标记，忽视了其固有的结构。此外，对比学习框架通常依赖大量困难的负样本，这对小型医学数据集而言不切实际。为应对这些挑战，我们提出了一种新的框架Adaptive Grouped Alignment (AGA)，用于从配对的医学图像和报告中捕捉结构化的语义。AGA引入了一种基于稀疏相似矩阵的双向分组机制。对于每张图像-报告配对，我们计算文本标记和图像 Patch 之间的细致相似性。每个标记选择与其最匹配的前m个 Patch 组成视觉分组，每个 Patch 选择与其最相关的标记组成语言分组。为了实现分组的自适应性，我们设计了两个阈值门控模块，分别称为语言分组阈值门和视觉分组阈值门，这些模块能够动态学习分组阈值。组表示通过相似性分数加权平均计算。为了使每个标记与其组表示对齐，我们引入了一种基于实例意识的组对齐损失，该损失在每张图像-文本配对内操作，无需外部负样本。最后，应用双向跨模态组对齐模块以增强视觉和语义组表示之间的细致对齐。在公共和私有数据集上的广泛实验表明，我们的方法在细调和零样本设置下的图像-文本检索和分类任务中均表现出强性能。 

---
# Reference-Guided Diffusion Inpainting For Multimodal Counterfactual Generation 

**Title (ZH)**: 参考指南扩散修复用于多模态反事实生成 

**Authors**: Alexandru Buburuzan  

**Link**: [PDF](https://arxiv.org/pdf/2507.23058)  

**Abstract**: Safety-critical applications, such as autonomous driving and medical image analysis, require extensive multimodal data for rigorous testing. Synthetic data methods are gaining prominence due to the cost and complexity of gathering real-world data, but they demand a high degree of realism and controllability to be useful. This work introduces two novel methods for synthetic data generation in autonomous driving and medical image analysis, namely MObI and AnydoorMed, respectively. MObI is a first-of-its-kind framework for Multimodal Object Inpainting that leverages a diffusion model to produce realistic and controllable object inpaintings across perceptual modalities, demonstrated simultaneously for camera and lidar. Given a single reference RGB image, MObI enables seamless object insertion into existing multimodal scenes at a specified 3D location, guided by a bounding box, while maintaining semantic consistency and multimodal coherence. Unlike traditional inpainting methods that rely solely on edit masks, this approach uses 3D bounding box conditioning to ensure accurate spatial positioning and realistic scaling. AnydoorMed extends this paradigm to the medical imaging domain, focusing on reference-guided inpainting for mammography scans. It leverages a diffusion-based model to inpaint anomalies with impressive detail preservation, maintaining the reference anomaly's structural integrity while semantically blending it with the surrounding tissue. Together, these methods demonstrate that foundation models for reference-guided inpainting in natural images can be readily adapted to diverse perceptual modalities, paving the way for the next generation of systems capable of constructing highly realistic, controllable and multimodal counterfactual scenarios. 

**Abstract (ZH)**: 安全关键应用（如自动驾驶和医学图像分析）需要大量的多模态数据进行严格的测试。合成数据方法由于收集真实世界数据的成本和复杂性逐渐受到重视，但它们需要极高的真实感和可控性才能发挥作用。本文介绍了两种新型的合成数据生成方法，分别为应用于自动驾驶的MObI和应用于医学图像分析的AnydoorMed。MObI是一种首创的多模态物体补全框架，利用扩散模型生成跨感知模态的真实且可控的物体补全，同时在相机和激光雷达观测量中进行展示。给定一张参考RGB图像，MObI能够根据bounding box指导，在指定的3D位置无缝插入物体，同时保持语义一致性和多模态一致性。不同于依赖于编辑掩模的传统补全方法，这种方法使用3D bounding box条件来确保准确的空间定位和真实的尺度。AnydoorMed将这一范式扩展到医学成像领域，重点关注参考引导的乳腺X光片补全。它利用基于扩散模型的方法以令人印象深刻的细节保真度补全异常，同时维护参考异常的结构完整性和与周围组织的语义融合。结合这些方法展示了，用于自然图像参考引导补全的基础模型可以很容易地适应各种感知模态，为下一代能够构建高度真实、可控和多模态反事实场景的系统铺平了道路。 

---
# Investigating the Invertibility of Multimodal Latent Spaces: Limitations of Optimization-Based Methods 

**Title (ZH)**: 探究多模态潜在空间的可逆性：基于优化的方法的局限性 

**Authors**: Siwoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2507.23010)  

**Abstract**: This paper investigates the inverse capabilities and broader utility of multimodal latent spaces within task-specific AI (Artificial Intelligence) models. While these models excel at their designed forward tasks (e.g., text-to-image generation, audio-to-text transcription), their potential for inverse mappings remains largely unexplored. We propose an optimization-based framework to infer input characteristics from desired outputs, applying it bidirectionally across Text-Image (BLIP, Flux.1-dev) and Text-Audio (Whisper-Large-V3, Chatterbox-TTS) modalities.
Our central hypothesis posits that while optimization can guide models towards inverse tasks, their multimodal latent spaces will not consistently support semantically meaningful and perceptually coherent inverse mappings. Experimental results consistently validate this hypothesis. We demonstrate that while optimization can force models to produce outputs that align textually with targets (e.g., a text-to-image model generating an image that an image captioning model describes correctly, or an ASR model transcribing optimized audio accurately), the perceptual quality of these inversions is chaotic and incoherent. Furthermore, when attempting to infer the original semantic input from generative models, the reconstructed latent space embeddings frequently lack semantic interpretability, aligning with nonsensical vocabulary tokens.
These findings highlight a critical limitation. multimodal latent spaces, primarily optimized for specific forward tasks, do not inherently possess the structure required for robust and interpretable inverse mappings. Our work underscores the need for further research into developing truly semantically rich and invertible multimodal latent spaces. 

**Abstract (ZH)**: 本文探讨了多功能潜空间在任务特定AI模型中的逆向能力和更广泛的应用潜力。虽然这些模型在设计的前向任务（如文本生成图像、音频转文本）上表现出色，但它们在逆向映射方面的潜力尚未得到充分探索。我们提出了一种基于优化的框架，用于从期望的输出推断输入特征，并将其应用于文本-图像（BLIP, Flux.1-dev）和文本-音频（Whisper-Large-V3, Chatterbox-TTS）模态的双向映射中。

我们的核心假设认为，虽然优化可以引导模型执行逆向任务，但其多功能潜空间并不总是能够支持语义上和感知上连贯的逆向映射。实验结果一致验证了这一假设。我们证明了虽然优化可以迫使模型生成与目标文本一致的内容（例如，一个文本生成图像的模型生成一个图像分类模型能正确描述的图像，或一个自动语音识别模型准确地转录优化后的音频），但这些逆向转换的感知质量是混沌且不连贯的。此外，当尝试从生成模型推断原始语义输入时，重构的潜空间嵌入往往缺乏语义可解释性，与无意义的词汇令牌一致。

这些发现指出了一个关键限制：主要用于特定前向任务优化的多功能潜空间，并不天生具备支持稳健且可解释的逆向映射的结构。我们的工作强调了进一步研究开发真正丰富语义且可逆的多功能潜空间的必要性。 

---
# Deep Learning Approaches for Multimodal Intent Recognition: A Survey 

**Title (ZH)**: 多模态意图识别的深度学习方法综述 

**Authors**: Jingwei Zhao, Yuhua Wen, Qifei Li, Minchi Hu, Yingying Zhou, Jingyao Xue, Junyang Wu, Yingming Gao, Zhengqi Wen, Jianhua Tao, Ya Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.22934)  

**Abstract**: Intent recognition aims to identify users' underlying intentions, traditionally focusing on text in natural language processing. With growing demands for natural human-computer interaction, the field has evolved through deep learning and multimodal approaches, incorporating data from audio, vision, and physiological signals. Recently, the introduction of Transformer-based models has led to notable breakthroughs in this domain. This article surveys deep learning methods for intent recognition, covering the shift from unimodal to multimodal techniques, relevant datasets, methodologies, applications, and current challenges. It provides researchers with insights into the latest developments in multimodal intent recognition (MIR) and directions for future research. 

**Abstract (ZH)**: 意图识别旨在识别用户背后的目的，传统上聚焦于自然语言处理中的文本。随着对自然人机交互需求的增加，该领域通过深度学习和多模态方法得到了发展， Incorporating 数据从音频、视觉和生理信号中获得。最近，基于Transformer的模型的引入在该领域取得了显著突破。本文综述了意图识别中的深度学习方法，涵盖了从单模态到多模态技术的转变、相关数据集、方法学、应用以及当前挑战。为研究人员提供了多模态意图识别（MIR）最新发展的洞察和未来研究方向。 

---
