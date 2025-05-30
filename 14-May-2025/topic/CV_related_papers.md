# Symbolically-Guided Visual Plan Inference from Uncurated Video Data 

**Title (ZH)**: 符号引导的视觉计划推断从未经整理的视频数据中 

**Authors**: Wenyan Yang, Ahmet Tikna, Yi Zhao, Yuying Zhang, Luigi Palopoli, Marco Roveri, Joni Pajarinen  

**Link**: [PDF](https://arxiv.org/pdf/2505.08444)  

**Abstract**: Visual planning, by offering a sequence of intermediate visual subgoals to a goal-conditioned low-level policy, achieves promising performance on long-horizon manipulation tasks. To obtain the subgoals, existing methods typically resort to video generation models but suffer from model hallucination and computational cost. We present Vis2Plan, an efficient, explainable and white-box visual planning framework powered by symbolic guidance. From raw, unlabeled play data, Vis2Plan harnesses vision foundation models to automatically extract a compact set of task symbols, which allows building a high-level symbolic transition graph for multi-goal, multi-stage planning. At test time, given a desired task goal, our planner conducts planning at the symbolic level and assembles a sequence of physically consistent intermediate sub-goal images grounded by the underlying symbolic representation. Our Vis2Plan outperforms strong diffusion video generation-based visual planners by delivering 53\% higher aggregate success rate in real robot settings while generating visual plans 35$\times$ faster. The results indicate that Vis2Plan is able to generate physically consistent image goals while offering fully inspectable reasoning steps. 

**Abstract (ZH)**: Visual规划：通过为基于目标的低级策略提供一系列中间视觉子目标，实现长期操作任务的优异性能。Vis2Plan：一种基于符号指导的高效可解释白盒视觉规划框架 

---
# MDF: Multi-Modal Data Fusion with CNN-Based Object Detection for Enhanced Indoor Localization Using LiDAR-SLAM 

**Title (ZH)**: 基于CNN对象检测的多模态数据融合以增强基于LiDAR-SLAM的室内定位 

**Authors**: Saqi Hussain Kalan, Boon Giin Lee, Wan-Young Chung  

**Link**: [PDF](https://arxiv.org/pdf/2505.08388)  

**Abstract**: Indoor localization faces persistent challenges in achieving high accuracy, particularly in GPS-deprived environments. This study unveils a cutting-edge handheld indoor localization system that integrates 2D LiDAR and IMU sensors, delivering enhanced high-velocity precision mapping, computational efficiency, and real-time adaptability. Unlike 3D LiDAR systems, it excels with rapid processing, low-cost scalability, and robust performance, setting new standards for emergency response, autonomous navigation, and industrial automation. Enhanced with a CNN-driven object detection framework and optimized through Cartographer SLAM (simultaneous localization and mapping ) in ROS, the system significantly reduces Absolute Trajectory Error (ATE) by 21.03%, achieving exceptional precision compared to state-of-the-art approaches like SC-ALOAM, with a mean x-position error of -0.884 meters (1.976 meters). The integration of CNN-based object detection ensures robustness in mapping and localization, even in cluttered or dynamic environments, outperforming existing methods by 26.09%. These advancements establish the system as a reliable, scalable solution for high-precision localization in challenging indoor scenarios 

**Abstract (ZH)**: 室内定位在实现高精度方面仍面临持久挑战，尤其是在GPS受限环境中。本研究揭示了一种集成2D LiDAR和IMU传感器的前沿手持室内定位系统，该系统提供了增强的高速精度制图、计算效率和实时适应性。与3D LiDAR系统相比，它在快速处理、低成本可扩展性和稳健性能方面表现出色，为应急响应、自主导航和工业自动化设立了新的标准。通过基于CNN的目标检测框架和ROS中的Cartographer SLAM优化，该系统显著降低了绝对轨迹误差（ATE）21.03%，其平均x位置误差为-0.884米（1.976米），优于诸如SC-ALOAM等最先进的方法。基于CNN的目标检测集成确保了在复杂或动态环境中具有鲁棒的制图和定位能力，比现有方法高出26.09%。这些进步确立了该系统在具有挑战性室内环境中的可靠且可扩展的高精度定位解决方案地位。 

---
# VISTA: Generative Visual Imagination for Vision-and-Language Navigation 

**Title (ZH)**: VISTA：视觉与语言导航中的生成性视觉想象 

**Authors**: Yanjia Huang, Mingyang Wu, Renjie Li, Zhengzhong Tu  

**Link**: [PDF](https://arxiv.org/pdf/2505.07868)  

**Abstract**: Vision-and-Language Navigation (VLN) tasks agents with locating specific objects in unseen environments using natural language instructions and visual cues. Many existing VLN approaches typically follow an 'observe-and-reason' schema, that is, agents observe the environment and decide on the next action to take based on the visual observations of their surroundings. They often face challenges in long-horizon scenarios due to limitations in immediate observation and vision-language modality gaps. To overcome this, we present VISTA, a novel framework that employs an 'imagine-and-align' navigation strategy. Specifically, we leverage the generative prior of pre-trained diffusion models for dynamic visual imagination conditioned on both local observations and high-level language instructions. A Perceptual Alignment Filter module then grounds these goal imaginations against current observations, guiding an interpretable and structured reasoning process for action selection. Experiments show that VISTA sets new state-of-the-art results on Room-to-Room (R2R) and RoboTHOR benchmarks, e.g.,+3.6% increase in Success Rate on R2R. Extensive ablation analysis underscores the value of integrating forward-looking imagination, perceptual alignment, and structured reasoning for robust navigation in long-horizon environments. 

**Abstract (ZH)**: Vision-and-Language Navigation (VLN)任务要求代理使用自然语言指令和视觉线索在未见过的环境中定位特定物体。许多现有的VLN方法通常遵循一种“观察与推理”模式，即代理观察环境并在基于周围视觉观察的基础上决定下一步采取的动作。它们在长期展望场景中常常面临挑战，这是因为即时观察的局限性和视觉语言模态之间的差距。为了克服这一问题，我们提出了一种新颖的VISTA框架，采用了一种“设想与对齐”的导航策略。具体而言，我们利用预训练扩散模型的生成先验，在结合局部观察和高层次语言指令的情况下进行动态视觉设想。然后，感知对齐滤波器模块将这些目标设想与当前观察相对接，指导可解释且结构化的推理过程以进行动作选择。实验结果显示，VISTA在Room-to-Room (R2R)和RoboTHOR基准测试中设立了新的最先进技术指标，例如在R2R中成功率为例，提高了3.6%。广泛的消融分析强调了融合前瞻想象、感知对齐和结构化推理对于在长期展望环境中实现鲁棒导航的价值。 

---
# DLO-Splatting: Tracking Deformable Linear Objects Using 3D Gaussian Splatting 

**Title (ZH)**: 基于3D高斯点云跟踪可变形线性对象 

**Authors**: Holly Dinkel, Marcel Büsching, Alberta Longhini, Brian Coltin, Trey Smith, Danica Kragic, Mårten Björkman, Timothy Bretl  

**Link**: [PDF](https://arxiv.org/pdf/2505.08644)  

**Abstract**: This work presents DLO-Splatting, an algorithm for estimating the 3D shape of Deformable Linear Objects (DLOs) from multi-view RGB images and gripper state information through prediction-update filtering. The DLO-Splatting algorithm uses a position-based dynamics model with shape smoothness and rigidity dampening corrections to predict the object shape. Optimization with a 3D Gaussian Splatting-based rendering loss iteratively renders and refines the prediction to align it with the visual observations in the update step. Initial experiments demonstrate promising results in a knot tying scenario, which is challenging for existing vision-only methods. 

**Abstract (ZH)**: DLO-Splatting: 一种基于多视图RGB图像和夹爪状态信息估计可变形线性对象3D形状的算法 

---
# MESSI: A Multi-Elevation Semantic Segmentation Image Dataset of an Urban Environment 

**Title (ZH)**: MESSI：城市环境多 elevation 语义分割图像数据集 

**Authors**: Barak Pinkovich, Boaz Matalon, Ehud Rivlin, Hector Rotstein  

**Link**: [PDF](https://arxiv.org/pdf/2505.08589)  

**Abstract**: This paper presents a Multi-Elevation Semantic Segmentation Image (MESSI) dataset comprising 2525 images taken by a drone flying over dense urban environments. MESSI is unique in two main features. First, it contains images from various altitudes, allowing us to investigate the effect of depth on semantic segmentation. Second, it includes images taken from several different urban regions (at different altitudes). This is important since the variety covers the visual richness captured by a drone's 3D flight, performing horizontal and vertical maneuvers. MESSI contains images annotated with location, orientation, and the camera's intrinsic parameters and can be used to train a deep neural network for semantic segmentation or other applications of interest (e.g., localization, navigation, and tracking). This paper describes the dataset and provides annotation details. It also explains how semantic segmentation was performed using several neural network models and shows several relevant statistics. MESSI will be published in the public domain to serve as an evaluation benchmark for semantic segmentation using images captured by a drone or similar vehicle flying over a dense urban environment. 

**Abstract (ZH)**: 多高度语义分割图像数据集（MESSI）：用于密集城市环境无人机飞行捕获图像的语义分割评估基准 

---
# A spherical amplitude-phase formulation for 3-D adaptive line-of-sight (ALOS) guidance with USGES stability guarantees 

**Title (ZH)**: 一种球面幅度相位公式在USGES稳定性保证下的3D自适应瞄准轴（ALOS）制导 

**Authors**: Erlend M. Coates, Thor I. Fossen  

**Link**: [PDF](https://arxiv.org/pdf/2505.08344)  

**Abstract**: A recently proposed 3-D adaptive line-of-sight (ALOS) path-following algorithm addressed coupled motion dynamics of marine craft, aircraft, and uncrewed vehicles under environmental disturbances such as wind, waves, and ocean currents. Stability analysis established uniform semiglobal exponential stability (USGES) of the cross- and vertical-track errors using a body-velocity-based amplitude-phase representation of the North-East-Down (NED) kinematic differential equations. In this brief paper, we revisit the ALOS framework and introduce a novel spherical amplitude-phase representation. This formulation yields a more geometrically intuitive and physically observable description of the guidance errors and enables a significantly simplified stability proof. Unlike the previous model, which relied on a vertical crab angle derived from body-frame velocities, the new representation uses an alternative vertical crab angle and retains the USGES property. It also removes restrictive assumptions such as constant altitude/depth or zero horizontal crab angle, and remains valid for general 3-D maneuvers with nonzero roll, pitch, and flight-path angles. 

**Abstract (ZH)**: 一种 Recently 提出的 3D 自适应视线路径跟踪算法在风、波浪和洋流等环境扰动下，处理了船舶、航空器和无人驾驶车辆的耦合运动动力学。稳定性分析利用基于体速度的幅度-相位表示的北东下（NED）运动微分方程，建立了交叉轨和垂直轨误差的均匀半全局指数稳定性（USGES）。本文重新审视了 ALOS 框架，并引入了一种新的球面幅度-相位表示。这种表示提供了更几何直观且物理上可观察的引导误差描述，并使得稳定性证明更为简化。与之前模型依赖于从体速度导出的垂直蟹角不同，新的表示使用了替代的垂直蟹角，并保留了 USGES 属性。此外，该表示消除了恒定高度/深度或零水平蟹角等限制假设，并适用于一般三维机动，其中滚转、俯仰和攻角均不为零。 

---
# Pose Estimation for Intra-cardiac Echocardiography Catheter via AI-Based Anatomical Understanding 

**Title (ZH)**: 基于AI解剖理解的经腔内超声心动图导管姿态估计 

**Authors**: Jaeyoung Huh, Ankur Kapoor, Young-Ho Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.07851)  

**Abstract**: Intra-cardiac Echocardiography (ICE) plays a crucial role in Electrophysiology (EP) and Structural Heart Disease (SHD) interventions by providing high-resolution, real-time imaging of cardiac structures. However, existing navigation methods rely on electromagnetic (EM) tracking, which is susceptible to interference and position drift, or require manual adjustments based on operator expertise. To overcome these limitations, we propose a novel anatomy-aware pose estimation system that determines the ICE catheter position and orientation solely from ICE images, eliminating the need for external tracking sensors. Our approach leverages a Vision Transformer (ViT)-based deep learning model, which captures spatial relationships between ICE images and anatomical structures. The model is trained on a clinically acquired dataset of 851 subjects, including ICE images paired with position and orientation labels normalized to the left atrium (LA) mesh. ICE images are patchified into 16x16 embeddings and processed through a transformer network, where a [CLS] token independently predicts position and orientation via separate linear layers. The model is optimized using a Mean Squared Error (MSE) loss function, balancing positional and orientational accuracy. Experimental results demonstrate an average positional error of 9.48 mm and orientation errors of (16.13 deg, 8.98 deg, 10.47 deg) across x, y, and z axes, confirming the model accuracy. Qualitative assessments further validate alignment between predicted and target views within 3D cardiac meshes. This AI-driven system enhances procedural efficiency, reduces operator workload, and enables real-time ICE catheter localization for tracking-free procedures. The proposed method can function independently or complement existing mapping systems like CARTO, offering a transformative approach to ICE-guided interventions. 

**Abstract (ZH)**: 基于ICE图像的解剖感知姿态估计系统在心内电生理和结构性心脏病介入中的应用 

---
# Arrow-Guided VLM: Enhancing Flowchart Understanding via Arrow Direction Encoding 

**Title (ZH)**: 箭头导向的VLM：通过箭头方向编码增强流程图理解 

**Authors**: Takamitsu Omasa, Ryo Koshihara, Masumi Morishige  

**Link**: [PDF](https://arxiv.org/pdf/2505.07864)  

**Abstract**: Flowcharts are indispensable tools in software design and business-process analysis, yet current vision-language models (VLMs) frequently misinterpret the directional arrows and graph topology that set these diagrams apart from natural images. We introduce a seven-stage pipeline grouped into three broader processes: (1) arrow-aware detection of nodes and arrow endpoints; (2) optical character recognition (OCR) to extract node text; and (3) construction of a structured prompt that guides the VLMs. Tested on a 90-question benchmark distilled from 30 annotated flowcharts, the method raises overall accuracy from 80 % to 89 % (+9 percentage points) without any task-specific fine-tuning. The gain is most pronounced for next-step queries (25/30 -> 30/30; 100 %, +17 pp); branch-result questions improve more modestly, and before-step questions remain difficult. A parallel evaluation with an LLM-as-a-Judge protocol shows the same trends, reinforcing the advantage of explicit arrow encoding. Limitations include dependence on detector and OCR precision, the small evaluation set, and residual errors at nodes with multiple incoming edges. Future work will enlarge the benchmark with synthetic and handwritten flowcharts and assess the approach on Business Process Model and Notation (BPMN) and Unified Modeling Language (UML). 

**Abstract (ZH)**: Flowcharts在软件设计和业务流程分析中的不可或缺工具，但当前的 vision-language 模型（VLMs）经常错误地解读这些图表中的方向箭头和图拓扑结构。我们介绍了由三个更广泛的处理过程组成的七阶段管道：(1) 具有箭头感知性的节点和箭头端点检测；(2) 光学字符识别（OCR）提取节点文本；(3) 构建结构化提示以指导VLMs。该方法在从30个标注的流程图中提取的90个问题基准测试中，从80%的准确率提高到89%（提高9个百分点），无需任何特定任务的微调。在下一步查询方面收益最为显著（25/30 -> 30/30；100%，+17个百分点）；分支结果查询有所改善，而前一步查询仍然困难。与基于LLM-as-a-Judge协议的并行评估显示了相同趋势，强化了明确箭头编码的优势。局限性包括对检测器和OCR精度的依赖、评价集规模较小以及节点有多条入边时残留的错误。未来的工作将进一步扩展基准测试，包括合成和手写流程图，并评估该方法在Business Process Model and Notation (BPMN)和Unified Modeling Language (UML)上的应用。 

---
# Advancing Food Nutrition Estimation via Visual-Ingredient Feature Fusion 

**Title (ZH)**: 通过视觉-ingredient特征融合促进食物营养估算 

**Authors**: Huiyan Qi, Bin Zhu, Chong-Wah Ngo, Jingjing Chen, Ee-Peng Lim  

**Link**: [PDF](https://arxiv.org/pdf/2505.08747)  

**Abstract**: Nutrition estimation is an important component of promoting healthy eating and mitigating diet-related health risks. Despite advances in tasks such as food classification and ingredient recognition, progress in nutrition estimation is limited due to the lack of datasets with nutritional annotations. To address this issue, we introduce FastFood, a dataset with 84,446 images across 908 fast food categories, featuring ingredient and nutritional annotations. In addition, we propose a new model-agnostic Visual-Ingredient Feature Fusion (VIF$^2$) method to enhance nutrition estimation by integrating visual and ingredient features. Ingredient robustness is improved through synonym replacement and resampling strategies during training. The ingredient-aware visual feature fusion module combines ingredient features and visual representation to achieve accurate nutritional prediction. During testing, ingredient predictions are refined using large multimodal models by data augmentation and majority voting. Our experiments on both FastFood and Nutrition5k datasets validate the effectiveness of our proposed method built in different backbones (e.g., Resnet, InceptionV3 and ViT), which demonstrates the importance of ingredient information in nutrition estimation. this https URL. 

**Abstract (ZH)**: 营养估计是促进健康饮食和减轻饮食相关健康风险的重要组成部分。尽管在食物分类和成分识别等方面取得了进步，但由于缺乏营养标注的数据集，营养估计的进步受到限制。为了解决这一问题，我们引入了FastFood数据集，该数据集包含84,446张图片，涵盖908个快速食品类别，并配有成分和营养标注。此外，我们提出了一种新的模型无关的视觉-成分特征融合（VIF²）方法，通过整合视觉和成分特征来增强营养估计。通过训练中的同义词替换和重采样策略提高了成分的鲁棒性。成分感知的视觉特征融合模块结合成分特征和视觉表示以实现准确的营养预测。在测试过程中，通过数据增强和多数投票策略，使用大型多模态模型 refining 成分预测。我们在FastFood和Nutrition5k数据集上的实验验证了我们所提出方法的有效性，该方法在不同骨干网络（例如，Resnet、InceptionV3和ViT）中构建，证明了成分信息在营养估计中的重要性。此链接：[原文链接] 

---
# Controllable Image Colorization with Instance-aware Texts and Masks 

**Title (ZH)**: 基于实例感知文本和掩码的可控图像着色 

**Authors**: Yanru An, Ling Gui, Qiang Hu, Chunlei Cai, Tianxiao Ye, Xiaoyun Zhang, Yanfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08705)  

**Abstract**: Recently, the application of deep learning in image colorization has received widespread attention. The maturation of diffusion models has further advanced the development of image colorization models. However, current mainstream image colorization models still face issues such as color bleeding and color binding errors, and cannot colorize images at the instance level. In this paper, we propose a diffusion-based colorization method MT-Color to achieve precise instance-aware colorization with use-provided guidance. To tackle color bleeding issue, we design a pixel-level mask attention mechanism that integrates latent features and conditional gray image features through cross-attention. We use segmentation masks to construct cross-attention masks, preventing pixel information from exchanging between different instances. We also introduce an instance mask and text guidance module that extracts instance masks and text representations of each instance, which are then fused with latent features through self-attention, utilizing instance masks to form self-attention masks to prevent instance texts from guiding the colorization of other areas, thus mitigating color binding errors. Furthermore, we apply a multi-instance sampling strategy, which involves sampling each instance region separately and then fusing the results. Additionally, we have created a specialized dataset for instance-level colorization tasks, GPT-color, by leveraging large visual language models on existing image datasets. Qualitative and quantitative experiments show that our model and dataset outperform previous methods and datasets. 

**Abstract (ZH)**: 基于扩散模型的精确实例感知颜色化方法MT-Color 

---
# A Survey of 3D Reconstruction with Event Cameras: From Event-based Geometry to Neural 3D Rendering 

**Title (ZH)**: 事件相机下的3D重建综述：从事件驱动几何到神经3D渲染 

**Authors**: Chuanzhi Xu, Haoxian Zhou, Langyi Chen, Haodong Chen, Ying Zhou, Vera Chung, Qiang Qu  

**Link**: [PDF](https://arxiv.org/pdf/2505.08438)  

**Abstract**: Event cameras have emerged as promising sensors for 3D reconstruction due to their ability to capture per-pixel brightness changes asynchronously. Unlike conventional frame-based cameras, they produce sparse and temporally rich data streams, which enable more accurate 3D reconstruction and open up the possibility of performing reconstruction in extreme environments such as high-speed motion, low light, or high dynamic range scenes. In this survey, we provide the first comprehensive review focused exclusively on 3D reconstruction using event cameras. The survey categorises existing works into three major types based on input modality - stereo, monocular, and multimodal systems, and further classifies them by reconstruction approach, including geometry-based, deep learning-based, and recent neural rendering techniques such as Neural Radiance Fields and 3D Gaussian Splatting. Methods with a similar research focus were organised chronologically into the most subdivided groups. We also summarise public datasets relevant to event-based 3D reconstruction. Finally, we highlight current research limitations in data availability, evaluation, representation, and dynamic scene handling, and outline promising future research directions. This survey aims to serve as a comprehensive reference and a roadmap for future developments in event-driven 3D reconstruction. 

**Abstract (ZH)**: 事件相机由于能够异步捕获逐像素亮度变化而成为了三维重建的有前途的传感器。本文综述首次全面聚焦于使用事件相机进行三维重建的研究。综述根据输入模态将现有工作划分为立体、单目和多模态系统三类，并进一步按重建方法分类，包括基于几何的方法、基于深度学习的方法以及最近的神经渲染技术如Neural Radiance Fields和3D Gaussian Splatting。具有类似研究重点的方法按照时间顺序进行了最详细的分类。此外，综述总结了与事件驱动三维重建相关的公开数据集，并指出现有研究在数据可用性、评估、表示以及动态场景处理方面的局限性，提出了有希望的未来研究方向。本文综述旨在成为事件驱动三维重建未来发展的全面参考和路线图。 

---
# A computer vision-based model for occupancy detection using low-resolution thermal images 

**Title (ZH)**: 基于计算机视觉的低分辨率热成像占用检测模型 

**Authors**: Xue Cui, Vincent Gbouna Zakka, Minhyun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.08336)  

**Abstract**: Occupancy plays an essential role in influencing the energy consumption and operation of heating, ventilation, and air conditioning (HVAC) systems. Traditional HVAC typically operate on fixed schedules without considering occupancy. Advanced occupant-centric control (OCC) adopted occupancy status in regulating HVAC operations. RGB images combined with computer vision (CV) techniques are widely used for occupancy detection, however, the detailed facial and body features they capture raise significant privacy concerns. Low-resolution thermal images offer a non-invasive solution that mitigates privacy issues. The study developed an occupancy detection model utilizing low-resolution thermal images and CV techniques, where transfer learning was applied to fine-tune the You Only Look Once version 5 (YOLOv5) model. The developed model ultimately achieved satisfactory performance, with precision, recall, mAP50, and mAP50 values approaching 1.000. The contributions of this model lie not only in mitigating privacy concerns but also in reducing computing resource demands. 

**Abstract (ZH)**: 占用状态在影响 Heating, Ventilation, and Air Conditioning (HVAC) 系统的能耗和运行中发挥着重要作用。传统 HVAC 通常基于固定时间表运行，不考虑占用状态。先进的以占用者为中心的控制（OCC）将占用状态纳入 HVAC 运行调控。RGB 图像结合计算机视觉（CV）技术广泛用于占用状态检测，但它们捕获的详细面部和身体特征引起了重大的隐私担忧。低分辨率热成像提供了一种非侵入性解决方案，可以缓解隐私问题。本研究开发了一种利用低分辨率热成像和 CV 技术的占用状态检测模型，其中应用了迁移学习对 You Only Look Once 版本 5 (YOLOv5) 模型进行了微调。所开发的模型最终取得了令人满意的效果，精度、召回率、mAP50 和 mAP50 值接近 1.000。该模型的贡献不仅在于缓解隐私问题，还在于减少计算资源需求。 

---
# M3G: Multi-Granular Gesture Generator for Audio-Driven Full-Body Human Motion Synthesis 

**Title (ZH)**: M3G：面向音频驱动全身人体运动合成的多粒度手势生成器 

**Authors**: Zhizhuo Yin, Yuk Hang Tsui, Pan Hui  

**Link**: [PDF](https://arxiv.org/pdf/2505.08293)  

**Abstract**: Generating full-body human gestures encompassing face, body, hands, and global movements from audio is a valuable yet challenging task in virtual avatar creation. Previous systems focused on tokenizing the human gestures framewisely and predicting the tokens of each frame from the input audio. However, one observation is that the number of frames required for a complete expressive human gesture, defined as granularity, varies among different human gesture patterns. Existing systems fail to model these gesture patterns due to the fixed granularity of their gesture tokens. To solve this problem, we propose a novel framework named Multi-Granular Gesture Generator (M3G) for audio-driven holistic gesture generation. In M3G, we propose a novel Multi-Granular VQ-VAE (MGVQ-VAE) to tokenize motion patterns and reconstruct motion sequences from different temporal granularities. Subsequently, we proposed a multi-granular token predictor that extracts multi-granular information from audio and predicts the corresponding motion tokens. Then M3G reconstructs the human gestures from the predicted tokens using the MGVQ-VAE. Both objective and subjective experiments demonstrate that our proposed M3G framework outperforms the state-of-the-art methods in terms of generating natural and expressive full-body human gestures. 

**Abstract (ZH)**: 基于音频生成涵盖面部、身体、双手和全局运动的全身人类手势：一个多粒度手势生成框架 

---
# Open the Eyes of MPNN: Vision Enhances MPNN in Link Prediction 

**Title (ZH)**: Open the Eyes of MPNN: Vision Enhances MPNN in Link Prediction 

**Authors**: Yanbin Wei, Xuehao Wang, Zhan Zhuang, Yang Chen, Shuhao Chen, Yulong Zhang, Yu Zhang, James Kwok  

**Link**: [PDF](https://arxiv.org/pdf/2505.08266)  

**Abstract**: Message-passing graph neural networks (MPNNs) and structural features (SFs) are cornerstones for the link prediction task. However, as a common and intuitive mode of understanding, the potential of visual perception has been overlooked in the MPNN community. For the first time, we equip MPNNs with vision structural awareness by proposing an effective framework called Graph Vision Network (GVN), along with a more efficient variant (E-GVN). Extensive empirical results demonstrate that with the proposed frameworks, GVN consistently benefits from the vision enhancement across seven link prediction datasets, including challenging large-scale graphs. Such improvements are compatible with existing state-of-the-art (SOTA) methods and GVNs achieve new SOTA results, thereby underscoring a promising novel direction for link prediction. 

**Abstract (ZH)**: 消息传递图神经网络和结构特征是链接预测任务的基石。然而，尽管视觉感知是一种常见且直观的方式，但在消息传递图神经网络社区中其潜力被忽视了。首次提出了一种有效框架Graph Vision Network (GVN)及其更高效的变体E-GVN，将视觉结构意识融入消息传递图神经网络中。广泛的经验结果表明，在七个链接预测数据集中，GVN一致地得益于视觉增强，包括具有挑战性的大规模图。这些改进与现有的最先进方法兼容，GVNs达到了新的最先进结果，从而强调了一个有希望的新方向用于链接预测。 

---
# Object detection in adverse weather conditions for autonomous vehicles using Instruct Pix2Pix 

**Title (ZH)**: 在恶劣天气条件下使用Instruct Pix2Pix进行自主车辆目标检测 

**Authors**: Unai Gurbindo, Axel Brando, Jaume Abella, Caroline König  

**Link**: [PDF](https://arxiv.org/pdf/2505.08228)  

**Abstract**: Enhancing the robustness of object detection systems under adverse weather conditions is crucial for the advancement of autonomous driving technology. This study presents a novel approach leveraging the diffusion model Instruct Pix2Pix to develop prompting methodologies that generate realistic datasets with weather-based augmentations aiming to mitigate the impact of adverse weather on the perception capabilities of state-of-the-art object detection models, including Faster R-CNN and YOLOv10. Experiments were conducted in two environments, in the CARLA simulator where an initial evaluation of the proposed data augmentation was provided, and then on the real-world image data sets BDD100K and ACDC demonstrating the effectiveness of the approach in real environments.
The key contributions of this work are twofold: (1) identifying and quantifying the performance gap in object detection models under challenging weather conditions, and (2) demonstrating how tailored data augmentation strategies can significantly enhance the robustness of these models. This research establishes a solid foundation for improving the reliability of perception systems in demanding environmental scenarios, and provides a pathway for future advancements in autonomous driving. 

**Abstract (ZH)**: 增强对象检测系统在不良天气条件下的稳健性对于自主驾驶技术的发展至关重要。本研究提出了一种新颖的方法，利用扩散模型Instruct Pix2Pix开发生成具有基于天气增强的现实数据集的提示方法，旨在减少不良天气对最先进的对象检测模型（包括Faster R-CNN和YOLOv10）感知能力的负面影响。实验在CARLA模拟器和真实世界数据集BDD100K及ACDC中进行，展示了该方法在实际环境中的有效性。本工作的主要贡献有两点：（1）识别并量化对象检测模型在恶劣天气条件下的性能差距，（2）演示定制的数据增强策略如何显著提高这些模型的稳健性。这项研究为在严峻环境场景中提高感知系统可靠性奠定了坚实基础，并为自主驾驶技术的未来发展提供了途径。 

---
# Computationally Efficient Diffusion Models in Medical Imaging: A Comprehensive Review 

**Title (ZH)**: 医学成像中计算高效扩散模型：综述 

**Authors**: Abdullah, Tao Huang, Ickjai Lee, Euijoon Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2505.07866)  

**Abstract**: The diffusion model has recently emerged as a potent approach in computer vision, demonstrating remarkable performances in the field of generative artificial intelligence. Capable of producing high-quality synthetic images, diffusion models have been successfully applied across a range of applications. However, a significant challenge remains with the high computational cost associated with training and generating these models. This study focuses on the efficiency and inference time of diffusion-based generative models, highlighting their applications in both natural and medical imaging. We present the most recent advances in diffusion models by categorizing them into three key models: the Denoising Diffusion Probabilistic Model (DDPM), the Latent Diffusion Model (LDM), and the Wavelet Diffusion Model (WDM). These models play a crucial role in medical imaging, where producing fast, reliable, and high-quality medical images is essential for accurate analysis of abnormalities and disease diagnosis. We first investigate the general framework of DDPM, LDM, and WDM and discuss the computational complexity gap filled by these models in natural and medical imaging. We then discuss the current limitations of these models as well as the opportunities and future research directions in medical imaging. 

**Abstract (ZH)**: 扩散模型 recently emerged as a powerful approach in计算机视觉，展示了生成人工智能领域令人瞩目的性能。能够生成高质量的合成图像，扩散模型已成功应用于多种应用中。然而，与训练和生成这些模型相关的高计算成本仍是一项重大挑战。本研究着眼于基于扩散的生成模型的效率和推断时间，强调其在自然和医学成像中的应用。我们通过将这些模型分类为三种关键模型——去噪扩散概率模型（DDPM）、潜在扩散模型（LDM）和小波扩散模型（WDM）——来概述最近在扩散模型方面的进展。这些模型在医学成像中扮演着重要角色，因为快速、可靠且高质量的医学图像生产对于准确分析异常和疾病诊断至关重要。我们首先探讨了DDPM、LDM和WDM的通用框架，并讨论了这些模型在自然和医学成像中填补的计算复杂性差距。然后，我们讨论了这些模型当前的局限性以及在医学成像中的机会和未来的研究方向。 

---
# Sub-diffraction terahertz backpropagation compressive imaging 

**Title (ZH)**: 亚衍射极限太赫兹反传播压缩成像 

**Authors**: Yongsheng Zhu, Shaojing Liu, Ximiao Wang, Runli Li, Haili Yang, Jiali Wang, Hongjia Zhu, Yanlin Ke, Ningsheng Xu, Huanjun Chen, Shaozhi Deng  

**Link**: [PDF](https://arxiv.org/pdf/2505.07839)  

**Abstract**: Terahertz single-pixel imaging (TSPI) has garnered significant attention due to its simplicity and cost-effectiveness. However, the relatively long wavelength of THz waves limits sub-diffraction-scale imaging resolution. Although TSPI technique can achieve sub-wavelength resolution, it requires harsh experimental conditions and time-consuming processes. Here, we propose a sub-diffraction THz backpropagation compressive imaging technique. We illuminate the object with monochromatic continuous-wave THz radiation. The transmitted THz wave is modulated by prearranged patterns generated on the back surface of a 500-{\mu}m-thick silicon wafer, realized through photoexcited carriers using a 532-nm laser. The modulated THz wave is then recorded by a single-element detector. An untrained neural network is employed to iteratively reconstruct the object image with an ultralow compression ratio of 1.5625% under a physical model constraint, thus reducing the long sampling times. To further suppress the diffraction-field effects, embedded with the angular spectrum propagation (ASP) theory to model the diffraction of THz waves during propagation, the network retrieves near-field information from the object, enabling sub-diffraction imaging with a spatial resolution of ~{\lambda}0/7 ({\lambda}0 = 833.3 {\mu}m at 0.36 THz) and eliminating the need for ultrathin photomodulators. This approach provides an efficient solution for advancing THz microscopic imaging and addressing other inverse imaging challenges. 

**Abstract (ZH)**: 太赫兹单像素成像 (TSPI) 由于其简单性和经济性而引起了广泛关注。然而，太赫兹波相对较长的波长限制了其亚衍射分辨能力。尽管TSPI技术可以实现亚波长分辨率，但需要严苛的实验条件和耗时的过程。在此，我们提出了一种亚衍射太赫兹反向传播压缩成像技术。我们用单色连续波太赫兹辐射照射物体。透过硅片背面预设模式调制的太赫兹波由532 nm激光激发载流子实现。调制后的太赫兹波由单像素探测器记录。在物理模型约束下，使用未训练的神经网络以超低压缩比（1.5625%）迭代重构物体图像，从而减少长时间的采样。为进一步抑制衍射场效应，结合入射角谱传播（ASP）理论来建模太赫兹波传播过程中的衍射，网络从物体中检索近场信息，从而实现约λ0/7（λ0 = 833.3 μm，频率为0.36 THz）的空间分辨率的亚衍射成像，无需超薄光调制器。该方法为推进太赫兹显微成像和解决其他逆向成像挑战提供了有效解决方案。 

---
