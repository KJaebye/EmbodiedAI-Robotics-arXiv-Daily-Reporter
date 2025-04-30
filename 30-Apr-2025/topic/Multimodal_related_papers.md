# mrCAD: Multimodal Refinement of Computer-aided Designs 

**Title (ZH)**: mrCAD: 多模态计算机辅助设计 refinement 

**Authors**: William P. McCarthy, Saujas Vaduguru, Karl D. D. Willis, Justin Matejka, Judith E. Fan, Daniel Fried, Yewen Pu  

**Link**: [PDF](https://arxiv.org/pdf/2504.20294)  

**Abstract**: A key feature of human collaboration is the ability to iteratively refine the concepts we have communicated. In contrast, while generative AI excels at the \textit{generation} of content, it often struggles to make specific language-guided \textit{modifications} of its prior outputs. To bridge the gap between how humans and machines perform edits, we present mrCAD, a dataset of multimodal instructions in a communication game. In each game, players created computer aided designs (CADs) and refined them over several rounds to match specific target designs. Only one player, the Designer, could see the target, and they must instruct the other player, the Maker, using text, drawing, or a combination of modalities. mrCAD consists of 6,082 communication games, 15,163 instruction-execution rounds, played between 1,092 pairs of human players. We analyze the dataset and find that generation and refinement instructions differ in their composition of drawing and text. Using the mrCAD task as a benchmark, we find that state-of-the-art VLMs are better at following generation instructions than refinement instructions. These results lay a foundation for analyzing and modeling a multimodal language of refinement that is not represented in previous datasets. 

**Abstract (ZH)**: 人类协作的关键特征在于不断迭代细化沟通中的概念。相比之下，虽然生成型AI在内容生成方面表现出色，但在特定语言引导的内容修改方面往往存在困难。为弥合人类和机器编辑性能之间的差距，我们呈现了mrCAD数据集，该数据集包含多模态指令在通信游戏中的应用。在每场游戏中，参与者创建了计算机辅助设计(CAD)并进行了多次迭代以匹配特定的目标设计。只有设计师能够看到目标，他们必须使用文本、绘图或多种模态组合来指导另一个玩家——制造者。mrCAD包含6,082场通信游戏，15,163轮指令执行，共计由1,092对人类玩家完成。我们分析该数据集发现，生成指令和细化指令在绘图和文本的组成上有差异。使用mrCAD任务作为基准，我们发现最先进的视觉语言模型更擅长遵循生成指令而非细化指令。这些结果为分析和建模先前数据集中未包含的多模态细化语言奠定了基础。 

---
# YoChameleon: Personalized Vision and Language Generation 

**Title (ZH)**: YoChameleon: 个性化视觉与语言生成 

**Authors**: Thao Nguyen, Krishna Kumar Singh, Jing Shi, Trung Bui, Yong Jae Lee, Yuheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.20998)  

**Abstract**: Large Multimodal Models (e.g., GPT-4, Gemini, Chameleon) have evolved into powerful tools with millions of users. However, they remain generic models and lack personalized knowledge of specific user concepts. Previous work has explored personalization for text generation, yet it remains unclear how these methods can be adapted to new modalities, such as image generation. In this paper, we introduce Yo'Chameleon, the first attempt to study personalization for large multimodal models. Given 3-5 images of a particular concept, Yo'Chameleon leverages soft-prompt tuning to embed subject-specific information to (i) answer questions about the subject and (ii) recreate pixel-level details to produce images of the subject in new contexts. Yo'Chameleon is trained with (i) a self-prompting optimization mechanism to balance performance across multiple modalities, and (ii) a ``soft-positive" image generation approach to enhance image quality in a few-shot setting. 

**Abstract (ZH)**: 大型多模态模型（如GPT-4、Gemini、Chameleon）已经演变成强大的工具，拥有数百万用户。然而，它们仍然是通用模型，缺乏特定用户概念的个性化知识。虽然以往的工作探讨了文本生成的个性化方法，但尚不清楚这些方法如何适应新的模态，如图像生成。在本文中，我们引入了Yo'Chameleon，这是首个研究大型多模态模型个性化的方法。给定特定概念的3-5张图片，Yo'Chameleon利用软提示调谐将主题特定的信息嵌入其中，以(i) 回答关于主题的问题和(ii) 重建像素级细节，从而在新上下文中生成主题的图像。Yo'Chameleon通过(i) 自我提示优化机制来平衡多模态下的性能，以及(ii) “软正面”图像生成方法在少样本设置中提升图像质量来进行训练。 

---
# SpaRE: Enhancing Spatial Reasoning in Vision-Language Models with Synthetic Data 

**Title (ZH)**: SpaRE: 通过合成数据增强视觉语言模型的空间推理能力 

**Authors**: Michael Ogezi, Freda Shi  

**Link**: [PDF](https://arxiv.org/pdf/2504.20648)  

**Abstract**: Vision-language models (VLMs) work well in tasks ranging from image captioning to visual question answering (VQA), yet they struggle with spatial reasoning, a key skill for understanding our physical world that humans excel at. We find that spatial relations are generally rare in widely used VL datasets, with only a few being well represented, while most form a long tail of underrepresented relations. This gap leaves VLMs ill-equipped to handle diverse spatial relationships. To bridge it, we construct a synthetic VQA dataset focused on spatial reasoning generated from hyper-detailed image descriptions in Localized Narratives, DOCCI, and PixMo-Cap. Our dataset consists of 455k samples containing 3.4 million QA pairs. Trained on this dataset, our Spatial-Reasoning Enhanced (SpaRE) VLMs show strong improvements on spatial reasoning benchmarks, achieving up to a 49% performance gain on the What's Up benchmark, while maintaining strong results on general tasks. Our work narrows the gap between human and VLM spatial reasoning and makes VLMs more capable in real-world tasks such as robotics and navigation. 

**Abstract (ZH)**: 视觉语言模型在从图像描述到视觉问答任务中表现出色，但在空间推理方面存在不足，这是人类擅长的关键技能。我们发现，广泛使用的VL数据集中空间关系普遍较稀少，只有少数关系被充分代表，大多数则形成了长尾分布的稀少关系。这一差距使得视觉语言模型在处理多样化的空间关系方面能力不足。为了弥合这一差距，我们构建了一个基于《局部叙述》、DOCCI和PixMo-Cap中的超详细图像描述生成的合成视觉问答数据集，专注于空间推理。该数据集包含455,000个样本，共计340万问答对。在该数据集上训练的增强空间推理的视觉语言模型在空间推理基准测试中表现出显著改进，在What's Up基准测试中取得了高达49%的性能提升，同时在一般任务上保持了强大的结果。我们的工作缩小了人类与视觉语言模型在空间推理方面的差距，并使视觉语言模型在诸如机器人技术和导航等实际任务中更加具备能力。 

---
# AlignDiT: Multimodal Aligned Diffusion Transformer for Synchronized Speech Generation 

**Title (ZH)**: AlignDiT: 多模态对齐扩散变换器用于同步语音生成 

**Authors**: Jeongsoo Choi, Ji-Hoon Kim, Kim Sung-Bin, Tae-Hyun Oh, Joon Son Chung  

**Link**: [PDF](https://arxiv.org/pdf/2504.20629)  

**Abstract**: In this paper, we address the task of multimodal-to-speech generation, which aims to synthesize high-quality speech from multiple input modalities: text, video, and reference audio. This task has gained increasing attention due to its wide range of applications, such as film production, dubbing, and virtual avatars. Despite recent progress, existing methods still suffer from limitations in speech intelligibility, audio-video synchronization, speech naturalness, and voice similarity to the reference speaker. To address these challenges, we propose AlignDiT, a multimodal Aligned Diffusion Transformer that generates accurate, synchronized, and natural-sounding speech from aligned multimodal inputs. Built upon the in-context learning capability of the DiT architecture, AlignDiT explores three effective strategies to align multimodal representations. Furthermore, we introduce a novel multimodal classifier-free guidance mechanism that allows the model to adaptively balance information from each modality during speech synthesis. Extensive experiments demonstrate that AlignDiT significantly outperforms existing methods across multiple benchmarks in terms of quality, synchronization, and speaker similarity. Moreover, AlignDiT exhibits strong generalization capability across various multimodal tasks, such as video-to-speech synthesis and visual forced alignment, consistently achieving state-of-the-art performance. The demo page is available at this https URL . 

**Abstract (ZH)**: 本文探讨了多模态到语音生成的任务，旨在从多种输入模态：文本、视频和参考语音中合成高质量的语音。由于其在电影制作、配音和虚拟化身等广泛应用中的潜力，该任务引起了越来越多的关注。尽管取得了近期进展，现有方法在语音清晰度、音频-视频同步、语音自然度以及语音与参考说话者音色相似度方面仍存在局限。为应对这些挑战，我们提出了AlignDiT，一种能够从对齐的多模态输入中生成准确、同步且自然的语音的多模态对齐扩散变换器。基于DiT架构的上下文学习能力，AlignDiT探索了三种有效的策略来对齐多模态表示。此外，我们引入了一种新颖的多模态去分类器自由引导机制，使模型在语音合成过程中能够自适应地平衡每种模态的信息。大量实验证明，AlignDiT在质量、同步性和说话者相似度方面显著优于现有方法，在多个基准测试中均取得了优异表现。AlignDiT在视频到语音合成和视觉强制对齐等多种多模态任务中均表现出强大的泛化能力，并且持续保持最佳性能。更多详情请参见此网页：![](this%20https%20URL)。 

---
# Weaving Context Across Images: Improving Vision-Language Models through Focus-Centric Visual Chains 

**Title (ZH)**: 在图像间编织语境：通过焦点为中心的视觉链增强视觉语言模型 

**Authors**: Juntian Zhang, Chuanqi cheng, Yuhan Liu, Wei Liu, Jian Luan, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2504.20199)  

**Abstract**: Vision-language models (VLMs) achieve remarkable success in single-image tasks. However, real-world scenarios often involve intricate multi-image inputs, leading to a notable performance decline as models struggle to disentangle critical information scattered across complex visual features. In this work, we propose Focus-Centric Visual Chain, a novel paradigm that enhances VLMs'perception, comprehension, and reasoning abilities in multi-image scenarios. To facilitate this paradigm, we propose Focus-Centric Data Synthesis, a scalable bottom-up approach for synthesizing high-quality data with elaborate reasoning paths. Through this approach, We construct VISC-150K, a large-scale dataset with reasoning data in the form of Focus-Centric Visual Chain, specifically designed for multi-image tasks. Experimental results on seven multi-image benchmarks demonstrate that our method achieves average performance gains of 3.16% and 2.24% across two distinct model architectures, without compromising the general vision-language capabilities. our study represents a significant step toward more robust and capable vision-language systems that can handle complex visual scenarios. 

**Abstract (ZH)**: Vision-language模型（VLMs）在单图像任务中取得了显著成功。然而，在现实场景中往往涉及复杂的多图像输入，导致模型在多图像场景中的性能显著下降，因为模型难以分离散布在复杂视觉特征中的关键信息。为此，我们提出了以聚焦为中心的视觉链（Focus-Centric Visual Chain）范式，该范式增强了模型在多图像场景中的感知、理解和推理能力。为了实现这一范式，我们提出了以聚焦为中心的数据合成方法，这是一种可扩展的自底向上的方法，用于合成具有详细推理路径的高质量数据。通过这种方法，我们构建了VISC-150K数据集，该数据集包含以聚焦为中心的视觉链形式的推理数据，专门设计用于多图像任务。实验结果显示，我们的方法在两种不同的模型架构上分别取得了3.16%和2.24%的平均性能提升，而不牺牲一般视觉语言能力。我们的研究代表了向更具鲁棒性和能力的视觉语言系统迈出的重要一步，这些系统能够处理复杂的视觉场景。 

---
