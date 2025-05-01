# CMD: Constraining Multimodal Distribution for Domain Adaptation in Stereo Matching 

**Title (ZH)**: CMD：约束多模态分布的领域适应立体匹配 

**Authors**: Zhelun Shen, Zhuo Li, Chenming Wu, Zhibo Rao, Lina Liu, Yuchao Dai, Liangjun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.21302)  

**Abstract**: Recently, learning-based stereo matching methods have achieved great improvement in public benchmarks, where soft argmin and smooth L1 loss play a core contribution to their success. However, in unsupervised domain adaptation scenarios, we observe that these two operations often yield multimodal disparity probability distributions in target domains, resulting in degraded generalization. In this paper, we propose a novel approach, Constrain Multi-modal Distribution (CMD), to address this issue. Specifically, we introduce \textit{uncertainty-regularized minimization} and \textit{anisotropic soft argmin} to encourage the network to produce predominantly unimodal disparity distributions in the target domain, thereby improving prediction accuracy. Experimentally, we apply the proposed method to multiple representative stereo-matching networks and conduct domain adaptation from synthetic data to unlabeled real-world scenes. Results consistently demonstrate improved generalization in both top-performing and domain-adaptable stereo-matching models. The code for CMD will be available at: \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 基于约束多模态分布的无监督场景适配立体匹配方法 

---
# Black-Box Visual Prompt Engineering for Mitigating Object Hallucination in Large Vision Language Models 

**Title (ZH)**: 黑盒视觉提示工程以减轻大型视觉语言模型中的物体幻象问题 

**Authors**: Sangmin Woo, Kang Zhou, Yun Zhou, Shuai Wang, Sheng Guan, Haibo Ding, Lin Lee Cheong  

**Link**: [PDF](https://arxiv.org/pdf/2504.21559)  

**Abstract**: Large Vision Language Models (LVLMs) often suffer from object hallucination, which undermines their reliability. Surprisingly, we find that simple object-based visual prompting -- overlaying visual cues (e.g., bounding box, circle) on images -- can significantly mitigate such hallucination; however, different visual prompts (VPs) vary in effectiveness. To address this, we propose Black-Box Visual Prompt Engineering (BBVPE), a framework to identify optimal VPs that enhance LVLM responses without needing access to model internals. Our approach employs a pool of candidate VPs and trains a router model to dynamically select the most effective VP for a given input image. This black-box approach is model-agnostic, making it applicable to both open-source and proprietary LVLMs. Evaluations on benchmarks such as POPE and CHAIR demonstrate that BBVPE effectively reduces object hallucination. 

**Abstract (ZH)**: 基于黑盒视觉提示工程的大型视觉语言模型对象幻觉缓解 

---
# GarmentDiffusion: 3D Garment Sewing Pattern Generation with Multimodal Diffusion Transformers 

**Title (ZH)**: GarmentDiffusion: 多模态扩散变换器驱动的3D服装缝制模板生成 

**Authors**: Xinyu Li, Qi Yao, Yuanda Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.21476)  

**Abstract**: Garment sewing patterns are fundamental design elements that bridge the gap between design concepts and practical manufacturing. The generative modeling of sewing patterns is crucial for creating diversified garments. However, existing approaches are limited either by reliance on a single input modality or by suboptimal generation efficiency. In this work, we present \textbf{\textit{GarmentDiffusion}}, a new generative model capable of producing centimeter-precise, vectorized 3D sewing patterns from multimodal inputs (text, image, and incomplete sewing pattern). Our method efficiently encodes 3D sewing pattern parameters into compact edge token representations, achieving a sequence length that is $\textbf{10}\times$ shorter than that of the autoregressive SewingGPT in DressCode. By employing a diffusion transformer, we simultaneously denoise all edge tokens along the temporal axis, while maintaining a constant number of denoising steps regardless of dataset-specific edge and panel statistics. With all combination of designs of our model, the sewing pattern generation speed is accelerated by $\textbf{100}\times$ compared to SewingGPT. We achieve new state-of-the-art results on DressCodeData, as well as on the largest sewing pattern dataset, namely GarmentCodeData. The project website is available at this https URL. 

**Abstract (ZH)**: 服装缝制图案是将设计理念与实际制造业连接起来的基本设计元素。缝制图案的生成模型对于创建多样化服装至关重要。然而，现有的方法要么依赖单一的输入模态，要么生成效率不佳。在本文中，我们提出了一种新的生成模型\textbf{\textit{GarmentDiffusion}}，该模型能够从多模态输入（文本、图像和不完整的缝制图案）生成厘米级精确的矢量3D缝制图案。我们的方法高效地将3D缝制图案参数编码为紧凑的边令牌表示，序列长度比DressCode中的自回归SewingGPT短10倍。通过使用扩散变压器，我们同时沿时间轴去噪所有边令牌，且去噪步骤数量始终保持不变，与数据集特定的边和面板统计数据无关。结合我们模型的所有设计组合，缝制图案生成速度比SewingGPT快100倍。我们在DressCodeData和最大的缝制图案数据集GarmentCodeData上均取得了新的state-of-the-art结果。项目网站可在以下链接访问。 

---
# DGFNet: End-to-End Audio-Visual Source Separation Based on Dynamic Gating Fusion 

**Title (ZH)**: DGFNet：基于动态门控融合的端到端音频-视觉源分离 

**Authors**: Yinfeng Yu, Shiyu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.21366)  

**Abstract**: Current Audio-Visual Source Separation methods primarily adopt two design strategies. The first strategy involves fusing audio and visual features at the bottleneck layer of the encoder, followed by processing the fused features through the decoder. However, when there is a significant disparity between the two modalities, this approach may lead to the loss of critical information. The second strategy avoids direct fusion and instead relies on the decoder to handle the interaction between audio and visual features. Nonetheless, if the encoder fails to integrate information across modalities adequately, the decoder may be unable to effectively capture the complex relationships between them. To address these issues, this paper proposes a dynamic fusion method based on a gating mechanism that dynamically adjusts the modality fusion degree. This approach mitigates the limitations of solely relying on the decoder and facilitates efficient collaboration between audio and visual features. Additionally, an audio attention module is introduced to enhance the expressive capacity of audio features, thereby further improving model performance. Experimental results demonstrate that our method achieves significant performance improvements on two benchmark datasets, validating its effectiveness and advantages in Audio-Visual Source Separation tasks. 

**Abstract (ZH)**: 当前的音频-视觉源分离方法主要采用两种设计策略。第一种策略是在编码器的瓶颈层融合音频和视觉特征，然后通过解码器处理融合后的特征。然而，当两种模态之间存在显著差异时，这种做法可能导致关键信息的丢失。第二种策略避免直接融合，而是依靠解码器来处理音频和视觉特征之间的交互。然而，如果编码器未能充分整合跨模态的信息，解码器可能无法有效捕捉它们之间的复杂关系。为了解决这些问题，本文提出了一种基于门控机制的动态融合方法，该方法动态调整模态融合程度。该方法减轻了仅依赖解码器的局限性，促进了音频和视觉特征的有效协作。此外，引入了音频注意力模块以增强音频特征的表达能力，进而进一步提高模型性能。实验结果表明，该方法在两个基准数据集上实现了显著的性能提升，验证了其在音频-视觉源分离任务中的有效性和优势。 

---
# Nexus-Gen: A Unified Model for Image Understanding, Generation, and Editing 

**Title (ZH)**: Nexus-Gen：统一的图像理解、生成和编辑模型 

**Authors**: Hong Zhang, Zhongjie Duan, Xingjun Wang, Yingda Chen, Yuze Zhao, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.21356)  

**Abstract**: Unified multimodal large language models (MLLMs) aim to integrate multimodal understanding and generation abilities through a single framework. Despite their versatility, existing open-source unified models exhibit performance gaps against domain-specific architectures. To bridge this gap, we present Nexus-Gen, a unified model that synergizes the language reasoning capabilities of LLMs with the image synthesis power of diffusion models. To align the embedding space of the LLM and diffusion model, we conduct a dual-phase alignment training process. (1) The autoregressive LLM learns to predict image embeddings conditioned on multimodal inputs, while (2) the vision decoder is trained to reconstruct high-fidelity images from these embeddings. During training the LLM, we identified a critical discrepancy between the autoregressive paradigm's training and inference phases, where error accumulation in continuous embedding space severely degrades generation quality. To avoid this issue, we introduce a prefilled autoregression strategy that prefills input sequence with position-embedded special tokens instead of continuous embeddings. Through dual-phase training, Nexus-Gen has developed the integrated capability to comprehensively address the image understanding, generation and editing tasks. All models, datasets, and codes are published at this https URL to facilitate further advancements across the field. 

**Abstract (ZH)**: 统一多模态大型语言模型（MLLMs）旨在通过单一框架集成多模态的理解和生成能力。为了弥合这一差距，我们提出了Nexus-Gen模型，该模型结合了大型语言模型的语言推理能力和扩散模型的图像合成能力。为了使语言模型和扩散模型的嵌入空间保持一致，我们采用了双重训练过程。在训练语言模型时，我们发现自回归范式的训练和推理阶段之间存在关键差异，连续嵌入空间中的误差积累严重降低了生成质量。为了避免这一问题，我们引入了预填充自回归策略，使用位置嵌入的特殊标记来填充输入序列，而不是连续嵌入。通过双重训练，Nexus-Gen具备了全面解决图像理解和生成编辑任务的集成能力。所有模型、数据集和代码均在此处发布，以促进该领域进一步的发展。 

---
# MemeBLIP2: A novel lightweight multimodal system to detect harmful memes 

**Title (ZH)**: MemeBLIP2: 一种新的轻量级多模态有害梗检测系统 

**Authors**: Jiaqi Liu, Ran Tong, Aowei Shen, Shuzheng Li, Changlin Yang, Lisha Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.21226)  

**Abstract**: Memes often merge visuals with brief text to share humor or opinions, yet some memes contain harmful messages such as hate speech. In this paper, we introduces MemeBLIP2, a light weight multimodal system that detects harmful memes by combining image and text features effectively. We build on previous studies by adding modules that align image and text representations into a shared space and fuse them for better classification. Using BLIP-2 as the core vision-language model, our system is evaluated on the PrideMM datasets. The results show that MemeBLIP2 can capture subtle cues in both modalities, even in cases with ironic or culturally specific content, thereby improving the detection of harmful material. 

**Abstract (ZH)**: memes often merge visuals with brief text to share humor or opinions, yet some memes contain harmful messages such as hate speech. In this paper, we introduce MemeBLIP2, a lightweight multimodal system that detects harmful memes by effectively combining image and text features. We build on previous studies by adding modules that align image and text representations into a shared space and fuse them for better classification. Using BLIP-2 as the core vision-language model, our system is evaluated on the PrideMM dataset. The results show that MemeBLIP2 can capture subtle cues in both modalities, even in cases with ironic or culturally specific content, thereby improving the detection of harmful material. 

---
# AGATE: Stealthy Black-box Watermarking for Multimodal Model Copyright Protection 

**Title (ZH)**: AGATE：多模态模型版权保护的隐蔽黑盒水印方案 

**Authors**: Jianbo Gao, Keke Gai, Jing Yu, Liehuang Zhu, Qi Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.21044)  

**Abstract**: Recent advancement in large-scale Artificial Intelligence (AI) models offering multimodal services have become foundational in AI systems, making them prime targets for model theft. Existing methods select Out-of-Distribution (OoD) data as backdoor watermarks and retrain the original model for copyright protection. However, existing methods are susceptible to malicious detection and forgery by adversaries, resulting in watermark evasion. In this work, we propose Model-\underline{ag}nostic Black-box Backdoor W\underline{ate}rmarking Framework (AGATE) to address stealthiness and robustness challenges in multimodal model copyright protection. Specifically, we propose an adversarial trigger generation method to generate stealthy adversarial triggers from ordinary dataset, providing visual fidelity while inducing semantic shifts. To alleviate the issue of anomaly detection among model outputs, we propose a post-transform module to correct the model output by narrowing the distance between adversarial trigger image embedding and text embedding. Subsequently, a two-phase watermark verification is proposed to judge whether the current model infringes by comparing the two results with and without the transform module. Consequently, we consistently outperform state-of-the-art methods across five datasets in the downstream tasks of multimodal image-text retrieval and image classification. Additionally, we validated the robustness of AGATE under two adversarial attack scenarios. 

**Abstract (ZH)**: Recent进展在大规模多模态人工智能模型的黑盒后门水印保护框架（AGATE） 

---
