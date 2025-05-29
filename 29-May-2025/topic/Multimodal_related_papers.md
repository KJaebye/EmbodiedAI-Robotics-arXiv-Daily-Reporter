# Functional Matching of Logic Subgraphs: Beyond Structural Isomorphism 

**Title (ZH)**: 逻辑子图的功能匹配：超越结构性同构 

**Authors**: Ziyang Zheng, Kezhi Li, Zhengyuan Shi, Qiang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.21988)  

**Abstract**: Subgraph matching in logic circuits is foundational for numerous Electronic Design Automation (EDA) applications, including datapath optimization, arithmetic verification, and hardware trojan detection. However, existing techniques rely primarily on structural graph isomorphism and thus fail to identify function-related subgraphs when synthesis transformations substantially alter circuit topology. To overcome this critical limitation, we introduce the concept of functional subgraph matching, a novel approach that identifies whether a given logic function is implicitly present within a larger circuit, irrespective of structural variations induced by synthesis or technology mapping. Specifically, we propose a two-stage multi-modal framework: (1) learning robust functional embeddings across AIG and post-mapping netlists for functional subgraph detection, and (2) identifying fuzzy boundaries using a graph segmentation approach. Evaluations on standard benchmarks (ITC99, OpenABCD, ForgeEDA) demonstrate significant performance improvements over existing structural methods, with average $93.8\%$ accuracy in functional subgraph detection and a dice score of $91.3\%$ in fuzzy boundary identification. 

**Abstract (ZH)**: 逻辑电路中的子图匹配是许多电子设计自动化(EDA)应用的基础，包括数据路径优化、算术验证和硬件木马检测。然而，现有技术主要依赖于结构性图同构，因此在综合变换显著改变电路拓扑时，无法识别功能相关的子图。为克服这一关键局限，我们引入了功能子图匹配的概念，这是一种新的方法，可以在忽略合成或技术映射诱导的结构性变化的情况下，确定给定逻辑函数是否隐含地存在于更大的电路中。具体地，我们提出了一种两阶段多模态框架：（1）学习AIG和后映射网表之间鲁棒的功能嵌入以进行功能子图检测；（2）使用图分割方法识别模糊边界。标准基准（ITC99、OpenABCD、ForgeEDA）上的评估证明了相对于现有结构性方法的显著性能提升，在功能子图检测中的平均准确率为93.8%，模糊边界识别的骰子分数为91.3%。 

---
# Spatial Knowledge Graph-Guided Multimodal Synthesis 

**Title (ZH)**: 空间知识图谱引导的多模态合成 

**Authors**: Yida Xue, Zhen Bi, Jinnan Yang, Jungang Lou, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.22633)  

**Abstract**: Recent advances in multimodal large language models (MLLMs) have significantly enhanced their capabilities; however, their spatial perception abilities remain a notable limitation. To address this challenge, multimodal data synthesis offers a promising solution. Yet, ensuring that synthesized data adhere to spatial common sense is a non-trivial task. In this work, we introduce SKG2Data, a novel multimodal synthesis approach guided by spatial knowledge graphs, grounded in the concept of knowledge-to-data generation. SKG2Data automatically constructs a Spatial Knowledge Graph (SKG) to emulate human-like perception of spatial directions and distances, which is subsequently utilized to guide multimodal data synthesis. Extensive experiments demonstrate that data synthesized from diverse types of spatial knowledge, including direction and distance, not only enhance the spatial perception and reasoning abilities of MLLMs but also exhibit strong generalization capabilities. We hope that the idea of knowledge-based data synthesis can advance the development of spatial intelligence. 

**Abstract (ZH)**: Recent Advances in Multimodal Large Language Models (MLLMs) Have Significantly Enhanced Their Capabilities; However, Their Spatial Perception Abilities Remain a Notable Limitation. To Address This Challenge, Multimodal Data Synthesis Offers a Promising Solution. Yet, Ensuring that Synthesized Data Adhere to Spatial Common Sense Is a Non-Trivial Task. In This Work, We Introduce SKG2Data, a Novel Multimodal Synthesis Approach Guided by Spatial Knowledge Graphs, Grounded in the Concept of Knowledge-to-Data Generation. SKG2Data Automatically Constructs a Spatial Knowledge Graph (SKG) to Emulate Human-Like Perception of Spatial Directions and Distances, Which Is Subsequently Utilized to Guide Multimodal Data Synthesis. Extensive Experiments Demonstrate That Data Synthesized from Diverse Types of Spatial Knowledge, Including Direction and Distance, Not Only Enhance the Spatial Perception and Reasoning Abilities of MLLMs but Also Exhibit Strong Generalization Capabilities. We Hope That the Idea of Knowledge-Based Data Synthesis Can Advance the Development of Spatial Intelligence. 

---
# RICO: Improving Accuracy and Completeness in Image Recaptioning via Visual Reconstruction 

**Title (ZH)**: RICO：通过视觉重建提高图像重述准确性与完整性 

**Authors**: Yuchi Wang, Yishuo Cai, Shuhuai Ren, Sihan Yang, Linli Yao, Yuanxin Liu, Yuanxing Zhang, Pengfei Wan, Xu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.22613)  

**Abstract**: Image recaptioning is widely used to generate training datasets with enhanced quality for various multimodal tasks. Existing recaptioning methods typically rely on powerful multimodal large language models (MLLMs) to enhance textual descriptions, but often suffer from inaccuracies due to hallucinations and incompleteness caused by missing fine-grained details. To address these limitations, we propose RICO, a novel framework that refines captions through visual reconstruction. Specifically, we leverage a text-to-image model to reconstruct a caption into a reference image, and prompt an MLLM to identify discrepancies between the original and reconstructed images to refine the caption. This process is performed iteratively, further progressively promoting the generation of more faithful and comprehensive descriptions. To mitigate the additional computational cost induced by the iterative process, we introduce RICO-Flash, which learns to generate captions like RICO using DPO. Extensive experiments demonstrate that our approach significantly improves caption accuracy and completeness, outperforms most baselines by approximately 10% on both CapsBench and CompreCap. Code released at this https URL. 

**Abstract (ZH)**: 图像重撰风气用于生成用于各种多模态任务的高质量训练数据集。现有的重撰方法通常依赖于强大的多模态大型语言模型（MLLMs）来增强文本描述，但往往会因为幻觉和由于细节信息缺失导致的不完整而受到影响。为了解决这些限制，我们提出了一种名为RICO的新框架，通过视觉重建来 refine 描述。具体而言，我们利用文本转图像模型将描述重构为参考图像，并提示MLLM识别原始图像和重构图像之间的差异以 refine 描述。该过程是迭代进行的，进一步促进生成更为忠实和全面的描述。为了缓解迭代过程引起的额外计算成本，我们引入了RICO-Flash，它利用DPO学习使用与RICO类似的方式生成描述。大量实验表明，我们的方法显著提高了描述的准确性和完整性，在CapsBench和CompreCap上分别比大多数基线方法高出约10%。代码发布在该网址。 

---
# Thinking with Generated Images 

**Title (ZH)**: 生成图像的思考 

**Authors**: Ethan Chern, Zhulin Hu, Steffi Chern, Siqi Kou, Jiadi Su, Yan Ma, Zhijie Deng, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.22525)  

**Abstract**: We present Thinking with Generated Images, a novel paradigm that fundamentally transforms how large multimodal models (LMMs) engage with visual reasoning by enabling them to natively think across text and vision modalities through spontaneous generation of intermediate visual thinking steps. Current visual reasoning with LMMs is constrained to either processing fixed user-provided images or reasoning solely through text-based chain-of-thought (CoT). Thinking with Generated Images unlocks a new dimension of cognitive capability where models can actively construct intermediate visual thoughts, critique their own visual hypotheses, and refine them as integral components of their reasoning process. We demonstrate the effectiveness of our approach through two complementary mechanisms: (1) vision generation with intermediate visual subgoals, where models decompose complex visual tasks into manageable components that are generated and integrated progressively, and (2) vision generation with self-critique, where models generate an initial visual hypothesis, analyze its shortcomings through textual reasoning, and produce refined outputs based on their own critiques. Our experiments on vision generation benchmarks show substantial improvements over baseline approaches, with our models achieving up to 50% (from 38% to 57%) relative improvement in handling complex multi-object scenarios. From biochemists exploring novel protein structures, and architects iterating on spatial designs, to forensic analysts reconstructing crime scenes, and basketball players envisioning strategic plays, our approach enables AI models to engage in the kind of visual imagination and iterative refinement that characterizes human creative, analytical, and strategic thinking. We release our open-source suite at this https URL. 

**Abstract (ZH)**: 生成图像辅助思考：一种根本性改变大型多模态模型视觉推理方式的新范式 

---
# A Closer Look at Multimodal Representation Collapse 

**Title (ZH)**: 更深入探讨多模态表示崩溃问题 

**Authors**: Abhra Chaudhuri, Anjan Dutta, Tu Bui, Serban Georgescu  

**Link**: [PDF](https://arxiv.org/pdf/2505.22483)  

**Abstract**: We aim to develop a fundamental understanding of modality collapse, a recently observed empirical phenomenon wherein models trained for multimodal fusion tend to rely only on a subset of the modalities, ignoring the rest. We show that modality collapse happens when noisy features from one modality are entangled, via a shared set of neurons in the fusion head, with predictive features from another, effectively masking out positive contributions from the predictive features of the former modality and leading to its collapse. We further prove that cross-modal knowledge distillation implicitly disentangles such representations by freeing up rank bottlenecks in the student encoder, denoising the fusion-head outputs without negatively impacting the predictive features from either modality. Based on the above findings, we propose an algorithm that prevents modality collapse through explicit basis reallocation, with applications in dealing with missing modalities. Extensive experiments on multiple multimodal benchmarks validate our theoretical claims. Project page: this https URL. 

**Abstract (ZH)**: 我们旨在对模态崩溃这一最近观察到的经验现象进行基本的理解，模态崩溃是指在进行多模态融合训练的模型倾向于仅依赖于子集模态，而忽略其他模态。我们展示了当一个模态中的噪声特征通过融合头部共享的神经元与另一个模态的预测特征纠缠在一起时，会发生模态崩溃，这会屏蔽掉前一模态预测特征的积极贡献，导致该模态的崩溃。进一步证明，跨模态知识蒸馏通过在学生编码器中释放秩瓶颈隐式地解纠缠此类表示，同时不负面影响任一模态的预测特征，从而净化融合头部的输出。基于上述发现，我们提出了一种算法，通过显式的基底重新分配防止模态崩溃，并应用于处理缺失模态。在多个多模态基准上的广泛实验验证了我们的理论推断。项目页面：https://this-url。 

---
# Advancing Multimodal Reasoning via Reinforcement Learning with Cold Start 

**Title (ZH)**: 基于冷启动的强化学习促进多模态推理 

**Authors**: Lai Wei, Yuting Li, Kaipeng Zheng, Chen Wang, Yue Wang, Linghe Kong, Lichao Sun, Weiran Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.22334)  

**Abstract**: Recent advancements in large language models (LLMs) have demonstrated impressive chain-of-thought reasoning capabilities, with reinforcement learning (RL) playing a crucial role in this progress. While "aha moment" patterns--where models exhibit self-correction through reflection--are often attributed to emergent properties from RL, we first demonstrate that these patterns exist in multimodal LLMs (MLLMs) prior to RL training but may not necessarily correlate with improved reasoning performance. Building on these insights, we present a comprehensive study on enhancing multimodal reasoning through a two-stage approach: (1) supervised fine-tuning (SFT) as a cold start with structured chain-of-thought reasoning patterns, followed by (2) reinforcement learning via GRPO to further refine these capabilities. Our extensive experiments show that this combined approach consistently outperforms both SFT-only and RL-only methods across challenging multimodal reasoning benchmarks. The resulting models achieve state-of-the-art performance among open-source MLLMs at both 3B and 7B scales, with our 7B model showing substantial improvements over base models (e.g., 66.3 %$\rightarrow$73.4 % on MathVista, 62.9 %$\rightarrow$70.4 % on We-Math) and our 3B model achieving performance competitive with several 7B models. Overall, this work provides practical guidance for building advanced multimodal reasoning models. Our code is available at this https URL. 

**Abstract (ZH)**: 近期大型语言模型的进步展示了强大的链式思考推理能力，强化学习在这一进展中起到了关键作用。虽然“洞察时刻”模式——模型通过反思进行自我修正——通常被归因于来自强化学习的 emergent 属性，我们首先证明了这些模式在强化学习之前的多模态大型语言模型（MLLMs）中存在，但未必与推理性能的提高有必然联系。基于这些见解，我们提出了一种全面的方法来增强多模态推理，该方法采用两阶段策略：（1）监督微调 (SFT) 作为冷启动，使用结构化的链式思考推理模式，随后是（2）通过 GRPO 进行强化学习以进一步完善这些能力。我们广泛的经验表明，这种结合的方法在具有挑战性的多模态推理基准测试中一致优于仅使用 SFT 和仅使用 RL 的方法。生成的模型在开源 MLLMs 中实现了最先进的性能，其中我们的 7B 模型在 MathVista 和 We-Math 上显示出显著改进（例如，从 66.3% 到 73.4%，从 62.9% 到 70.4%），而我们的 3B 模型的性能与一些 7B 模型相当。总之，这项工作为构建高级多模态推理模型提供了实用指导。我们的代码可在以下网址获取。 

---
# Investigating Mechanisms for In-Context Vision Language Binding 

**Title (ZH)**: 探究上下文视知觉语言结合机制 

**Authors**: Darshana Saravanan, Makarand Tapaswi, Vineet Gandhi  

**Link**: [PDF](https://arxiv.org/pdf/2505.22200)  

**Abstract**: To understand a prompt, Vision-Language models (VLMs) must perceive the image, comprehend the text, and build associations within and across both modalities. For instance, given an 'image of a red toy car', the model should associate this image to phrases like 'car', 'red toy', 'red object', etc. Feng and Steinhardt propose the Binding ID mechanism in LLMs, suggesting that the entity and its corresponding attribute tokens share a Binding ID in the model activations. We investigate this for image-text binding in VLMs using a synthetic dataset and task that requires models to associate 3D objects in an image with their descriptions in the text. Our experiments demonstrate that VLMs assign a distinct Binding ID to an object's image tokens and its textual references, enabling in-context association. 

**Abstract (ZH)**: 视觉-语言模型需通过感知图像、理解文本并在两者之间建立联系来理解提示。例如，给定一幅“红色玩具汽车”的图像，模型应将该图像与“汽车”、“红色玩具”、“红色物体”等短语建立关联。冯和斯坦哈特在LLMs中提出了绑定ID机制，建议实体及其对应属性词令牌在模型激活中共享一个绑定ID。我们通过一个合成数据集和任务研究了这一机制在VLMs中的应用，该任务要求模型将图像中的3D对象与文本中的描述关联起来。实验结果表明，VLMs为图像中对象的词令牌及其文本引用分配了独特的绑定ID，从而实现上下文相关联。 

---
# Cross-modal RAG: Sub-dimensional Retrieval-Augmented Text-to-Image Generation 

**Title (ZH)**: 跨模态RAG：子维度检索增强文本到图像生成 

**Authors**: Mengdan Zhu, Senhao Cheng, Guangji Bai, Yifei Zhang, Liang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.21956)  

**Abstract**: Text-to-image generation increasingly demands access to domain-specific, fine-grained, and rapidly evolving knowledge that pretrained models cannot fully capture. Existing Retrieval-Augmented Generation (RAG) methods attempt to address this by retrieving globally relevant images, but they fail when no single image contains all desired elements from a complex user query. We propose Cross-modal RAG, a novel framework that decomposes both queries and images into sub-dimensional components, enabling subquery-aware retrieval and generation. Our method introduces a hybrid retrieval strategy - combining a sub-dimensional sparse retriever with a dense retriever - to identify a Pareto-optimal set of images, each contributing complementary aspects of the query. During generation, a multimodal large language model is guided to selectively condition on relevant visual features aligned to specific subqueries, ensuring subquery-aware image synthesis. Extensive experiments on MS-COCO, Flickr30K, WikiArt, CUB, and ImageNet-LT demonstrate that Cross-modal RAG significantly outperforms existing baselines in both retrieval and generation quality, while maintaining high efficiency. 

**Abstract (ZH)**: 跨模态RAG：分解查询和图像以实现亚查询意识的检索与生成 

---
# Towards Comprehensive Scene Understanding: Integrating First and Third-Person Views for LVLMs 

**Title (ZH)**: 面向全面场景理解：结合第一人称和第三人称视角的LVLMs研究 

**Authors**: Insu Lee, Wooje Park, Jaeyun Jang, Minyoung Noh, Kyuhong Shim, Byonghyo Shim  

**Link**: [PDF](https://arxiv.org/pdf/2505.21955)  

**Abstract**: Large vision-language models (LVLMs) are increasingly deployed in interactive applications such as virtual and augmented reality, where first-person (egocentric) view captured by head-mounted cameras serves as key input. While this view offers fine-grained cues about user attention and hand-object interactions, their narrow field of view and lack of global context often lead to failures on spatially or contextually demanding queries. To address this, we introduce a framework that augments egocentric inputs with third-person (exocentric) views, providing complementary information such as global scene layout and object visibility to LVLMs. We present E3VQA, the first benchmark for multi-view question answering with 4K high-quality question-answer pairs grounded in synchronized ego-exo image pairs. Additionally, we propose M3CoT, a training-free prompting technique that constructs a unified scene representation by integrating scene graphs from three complementary perspectives. M3CoT enables LVLMs to reason more effectively across views, yielding consistent performance gains (4.84% for GPT-4o and 5.94% for Gemini 2.0 Flash) over a recent CoT baseline. Our extensive evaluation reveals key strengths and limitations of LVLMs in multi-view reasoning and highlights the value of leveraging both egocentric and exocentric inputs. 

**Abstract (ZH)**: 基于第一人称和第三人称视角的大规模视觉-语言模型多视角问答框架 

---
# MMTBENCH: A Unified Benchmark for Complex Multimodal Table Reasoning 

**Title (ZH)**: MMTBENCH：统一的复杂多模态表推理基准测试 

**Authors**: Prasham Yatinkumar Titiya, Jainil Trivedi, Chitta Baral, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2505.21771)  

**Abstract**: Multimodal tables those that integrate semi structured data with visual elements such as charts and maps are ubiquitous across real world domains, yet they pose a formidable challenge to current vision language models (VLMs). While Large Language models (LLMs) and VLMs have demonstrated strong capabilities in text and image understanding, their performance on complex, real world multimodal table reasoning remains unexplored. To bridge this gap, we introduce MMTBENCH (Multimodal Table Benchmark), a benchmark consisting of 500 real world multimodal tables drawn from diverse real world sources, with a total of 4021 question answer pairs. MMTBENCH questions cover four question types (Explicit, Implicit, Answer Mention, and Visual Based), five reasoning types (Mathematical, Extrema Identification, Fact Verification, Vision Based, and Others), and eight table types (Single/Multiple Entity, Maps and Charts with Entities, Single/Multiple Charts, Maps, and Visualizations). Extensive evaluation of state of the art models on all types reveals substantial performance gaps, particularly on questions requiring visual-based reasoning and multi-step inference. These findings show the urgent need for improved architectures that more tightly integrate vision and language processing. By providing a challenging, high-quality resource that mirrors the complexity of real-world tasks, MMTBENCH underscores its value as a resource for future research on multimodal tables. 

**Abstract (ZH)**: 多模态表格基准：包含了来自多种真实世界来源的500个真实世界多模态表格，共计4021个问答对，包含四种问题类型、五种推理类型和八种表格类型。通过对最新模型的全面评估发现，在需要视觉推理和多步推理的问题上存在显著性能差距。这些发现表明，急需改进更紧密集成视觉和语言处理的架构。通过提供一个具有挑战性和高质量的资源，MMTBENCH突显了其作为未来研究多模态表格资源的价值。 

---
# OmniResponse: Online Multimodal Conversational Response Generation in Dyadic Interactions 

**Title (ZH)**: 全方位响应：双人互动中在线多模态对话响应生成 

**Authors**: Cheng Luo, Jianghui Wang, Bing Li, Siyang Song, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2505.21724)  

**Abstract**: In this paper, we introduce Online Multimodal Conversational Response Generation (OMCRG), a novel task that aims to online generate synchronized verbal and non-verbal listener feedback, conditioned on the speaker's multimodal input. OMCRG reflects natural dyadic interactions and poses new challenges in achieving synchronization between the generated audio and facial responses of the listener. To address these challenges, we innovatively introduce text as an intermediate modality to bridge the audio and facial responses. We hence propose OmniResponse, a Multimodal Large Language Model (MLLM) that autoregressively generates high-quality multi-modal listener responses. OmniResponse leverages a pretrained LLM enhanced with two novel components: Chrono-Text, which temporally anchors generated text tokens, and TempoVoice, a controllable online TTS module that produces speech synchronized with facial reactions. To support further OMCRG research, we present ResponseNet, a new dataset comprising 696 high-quality dyadic interactions featuring synchronized split-screen videos, multichannel audio, transcripts, and facial behavior annotations. Comprehensive evaluations conducted on ResponseNet demonstrate that OmniResponse significantly outperforms baseline models in terms of semantic speech content, audio-visual synchronization, and generation quality. 

**Abstract (ZH)**: 在线多模态对话响应生成（OMCRG）及其解决方法：OmniResponse模型的研究 

---
# Privacy-Preserving Chest X-ray Report Generation via Multimodal Federated Learning with ViT and GPT-2 

**Title (ZH)**: 基于ViT和GPT-2的多模态联邦学习的隐私保护胸片报告生成 

**Authors**: Md. Zahid Hossain, Mustofa Ahmed, Most. Sharmin Sultana Samu, Md. Rakibul Islam  

**Link**: [PDF](https://arxiv.org/pdf/2505.21715)  

**Abstract**: The automated generation of radiology reports from chest X-ray images holds significant promise in enhancing diagnostic workflows while preserving patient privacy. Traditional centralized approaches often require sensitive data transfer, posing privacy concerns. To address this, the study proposes a Multimodal Federated Learning framework for chest X-ray report generation using the IU-Xray dataset. The system utilizes a Vision Transformer (ViT) as the encoder and GPT-2 as the report generator, enabling decentralized training without sharing raw data. Three Federated Learning (FL) aggregation strategies: FedAvg, Krum Aggregation and a novel Loss-aware Federated Averaging (L-FedAvg) were evaluated. Among these, Krum Aggregation demonstrated superior performance across lexical and semantic evaluation metrics such as ROUGE, BLEU, BERTScore and RaTEScore. The results show that FL can match or surpass centralized models in generating clinically relevant and semantically rich radiology reports. This lightweight and privacy-preserving framework paves the way for collaborative medical AI development without compromising data confidentiality. 

**Abstract (ZH)**: 基于胸片图像的自动化放射报告生成在提升诊断工作流程的同时保护患者隐私方面具有重要潜力。传统的集中式方法常常需要传输敏感数据，存在隐私风险。为此，研究提出了一种用于生成胸片报告的多模态联邦学习框架，利用IU-Xray数据集。该系统采用Vision Transformer (ViT) 作为编码器和GPT-2作为报告生成器，实现无需共享原始数据的去中心化训练。三种联邦学习（FL）聚合策略：FedAvg、Krum聚合和一种新颖的损失感知联邦平均（L-FedAvg）进行了评估。其中，Krum聚合在ROUGE、BLEU、BERTScore和RaTEScore等词汇和语义评估指标上表现出更优性能。结果显示，FL能够在生成临床相关且语义丰富的放射报告方面与集中式模型相匹配甚至超越。这一轻量级且保护隐私的框架为无需牺牲数据保密性的协作医学AI开发铺平了道路。 

---
# Benign-to-Toxic Jailbreaking: Inducing Harmful Responses from Harmless Prompts 

**Title (ZH)**: 良性到有害的 Jailbreaking：无害提示诱导有害响应 

**Authors**: Hee-Seon Kim, Minbeom Kim, Wonjun Lee, Kihyun Kim, Changick Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.21556)  

**Abstract**: Optimization-based jailbreaks typically adopt the Toxic-Continuation setting in large vision-language models (LVLMs), following the standard next-token prediction objective. In this setting, an adversarial image is optimized to make the model predict the next token of a toxic prompt. However, we find that the Toxic-Continuation paradigm is effective at continuing already-toxic inputs, but struggles to induce safety misalignment when explicit toxic signals are absent. We propose a new paradigm: Benign-to-Toxic (B2T) jailbreak. Unlike prior work, we optimize adversarial images to induce toxic outputs from benign conditioning. Since benign conditioning contains no safety violations, the image alone must break the model's safety mechanisms. Our method outperforms prior approaches, transfers in black-box settings, and complements text-based jailbreaks. These results reveal an underexplored vulnerability in multimodal alignment and introduce a fundamentally new direction for jailbreak approaches. 

**Abstract (ZH)**: 基于优化的 Jailbreak 通常采用大型视觉-语言模型中的有毒连续输入设置（Toxic-Continuation），遵循标准的下一个 token 预测目标。在这种设置中，对抗图像被优化以使模型预测有毒提示的下一个 token。然而，我们发现，有毒连续输入范式在处理已经具有毒性输入时有效，但在缺乏明确的毒性信号时难以导致安全对齐失效。我们提出了一种新的范式：良性转有毒（B2T）Jailbreak。与以往工作不同，我们优化对抗图像以从良性的条件生成有毒输出。由于良性条件中不包含安全违规信号，图像本身必须打破模型的安全机制。我们的方法优于以往的方法，在黑盒设置中具有可迁移性，并补充了基于文本的 Jailbreak 方法。这些结果揭示了多模态对齐中未被充分探索的脆弱性，并引入了 Jailbreak 方法的新基本方向。 

---
# Image Tokens Matter: Mitigating Hallucination in Discrete Tokenizer-based Large Vision-Language Models via Latent Editing 

**Title (ZH)**: Image Tokens matters: 减轻基于离散标记器的大规模视觉-语言模型中幻觉影响的潜在编辑方法 

**Authors**: Weixing Wang, Zifeng Ding, Jindong Gu, Rui Cao, Christoph Meinel, Gerard de Melo, Haojin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.21547)  

**Abstract**: Large Vision-Language Models (LVLMs) with discrete image tokenizers unify multimodal representations by encoding visual inputs into a finite set of tokens. Despite their effectiveness, we find that these models still hallucinate non-existent objects. We hypothesize that this may be due to visual priors induced during training: When certain image tokens frequently co-occur in the same spatial regions and represent shared objects, they become strongly associated with the verbalizations of those objects. As a result, the model may hallucinate by evoking visually absent tokens that often co-occur with present ones. To test this assumption, we construct a co-occurrence graph of image tokens using a segmentation dataset and employ a Graph Neural Network (GNN) with contrastive learning followed by a clustering method to group tokens that frequently co-occur in similar visual contexts. We find that hallucinations predominantly correspond to clusters whose tokens dominate the input, and more specifically, that the visually absent tokens in those clusters show much higher correlation with hallucinated objects compared to tokens present in the image. Based on this observation, we propose a hallucination mitigation method that suppresses the influence of visually absent tokens by modifying latent image embeddings during generation. Experiments show our method reduces hallucinations while preserving expressivity. Code is available at this https URL 

**Abstract (ZH)**: 大视觉-语言模型中的离散图像标记器通过将视觉输入编码为有限的token集合来统一多模态表示。尽管这些模型效果显著，但我们发现它们仍然会产生不存在的物体。我们认为这可能是由于训练过程中引入的视觉先验：当某些图像token在相同的空间区域内频繁共现并表示共享物体时，它们会与这些物体的文本描述产生强烈的关联。因此，模型可能会通过唤起与现有关联但视觉上不存在的token来产生幻觉。为了验证这一假设，我们使用分割数据集构建图像token的共现图，并使用对比学习后的图神经网络（GNN）及聚类方法来分组在相似视觉上下文中频繁共现的tokens。我们发现，幻觉主要与支配输入的token群组相对应，更具体地说，这些群组中视觉上不存在的tokens与幻觉物体的相关性远高于图像中存在的tokens。基于这一观察，我们提出了一种通过修改生成过程中潜变量图像嵌入来抑制视觉上不存在的token影响的方法。实验表明，该方法能减少幻觉并保持表达能力。代码可在以下链接获取。 

---
# More Thinking, Less Seeing? Assessing Amplified Hallucination in Multimodal Reasoning Models 

**Title (ZH)**: 更多思考，更少观察？多模态推理模型中的放大幻觉评估 

**Authors**: Chengzhi Liu, Zhongxing Xu, Qingyue Wei, Juncheng Wu, James Zou, Xin Eric Wang, Yuyin Zhou, Sheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.21523)  

**Abstract**: Test-time compute has empowered multimodal large language models to generate extended reasoning chains, yielding strong performance on tasks such as multimodal math reasoning. However, this improved reasoning ability often comes with increased hallucination: as generations become longer, models tend to drift away from image-grounded content and rely more heavily on language priors. Attention analysis shows that longer reasoning chains lead to reduced focus on visual inputs, which contributes to hallucination. To systematically study this phenomenon, we introduce RH-AUC, a metric that quantifies how a model's perception accuracy changes with reasoning length, allowing us to evaluate whether the model preserves visual grounding during reasoning. We also release RH-Bench, a diagnostic benchmark that spans a variety of multimodal tasks, designed to assess the trade-off between reasoning ability and hallucination. Our analysis reveals that (i) larger models typically achieve a better balance between reasoning and perception, and (ii) this balance is influenced more by the types and domains of training data than by its overall volume. These findings underscore the importance of evaluation frameworks that jointly consider both reasoning quality and perceptual fidelity. 

**Abstract (ZH)**: Test-time计算使多模态大型语言模型能够生成扩展的推理链，从而在多模态数学推理等任务上表现出色。然而，这种增强的推理能力通常伴随着较高的幻觉率：随着生成变得越来越长，模型往往会偏离图像相关的内容，更多地依赖语言先验。注意力分析显示，较长的推理链会导致对视觉输入的关注度降低，从而导致幻觉。为系统地研究这一现象，我们引入了RH-AUC这一度量标准，量化模型感知准确度随推理长度的变化，以评估模型在推理过程中是否保留了视觉接地。我们还发布了RH-Bench这一诊断基准，涵盖多种多模态任务，旨在评估推理能力和幻觉之间的权衡。我们的分析表明，(i) 较大的模型通常在推理和感知之间取得了更好的平衡，(ii) 这种平衡更多地受训练数据类型和领域的影响，而不是数据总量的影响。这些发现强调了同时考虑推理质量和感知保真的评估框架的重要性。 

---
