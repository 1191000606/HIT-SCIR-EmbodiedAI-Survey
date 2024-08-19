# HIT-SCIR-EmbodiedAI-Survey

# Paper List

## Embodied Perception

### Object Perception

### Scene Perception

#### Information Acquisition and Composition

**Primary sources: visual, laser, and radar**

- Scene-aware learning network for radar object detection
- An ego-motion estimation method using millimeter-wave radar in 3D scene reconstruction
- A quality improvement method for 3D laser slam point clouds based on geometric primitives of the scan scene
- Multi-modal data-efficient 3d scene understanding for autonomous driving

**Alternative scene information sources**

- Sporadic Audio-Visual Embodied Assistive Robot Navigation For Human Tracking
- Look, listen, and act: Towards audio-visual embodied navigation
- StereoTac: A novel visuotactile sensor that combines tactile sensing with 3D vision
- Omnitact: A multi-directional high-resolution touch sensor
- Co-operative smell-based navigation for mobile robots
- Chemical sensing at the robot fingertips: Toward automated taste discrimination in food samples
- Scene recognition with infra-red, low-light, and sensor fused imagery
- Firefighting robot stereo infrared vision and radar sensor fusion for imaging through smoke
- Seeing Nearby 3D Scenes using Ultrasonic Sensors
- Indoor 3D reconstruction using camera, IMU and ultrasonic sensors

**Composition of scene information**

#### Scene reconstruction

- Simultaneous localization and mapping: part I
- Visual SLAM algorithms: A survey from 2010 to 2016

##### New Methods

- Toward geometric deep slam
- Cnn-slam: Real-time dense monocular slam with learned depth prediction
- Undeepvo: Monocular visual odometry through
- A survey of state-of-the-art on visual SLAM
- A comprehensive survey of visual slam algorithms
- An overview on visual slam: From tradition to semantic

##### New Task

**Active Mapping**

- Learning the next best view for 3d point clouds via topological features
- Bag of views: An appearance-based approach to next-best-view planning for 3d reconstruction
- Object-aware guidance for autonomous scene reconstruction
- Multi-robot collaborative dense scene reconstruction

**Active Localization**

- Extended Kalman filter-based mobile robot localization with intermittent measurements
- Markov localization for mobile robots in dynamic environments
- Monte carlo localization for mobile robots
- Active neural localization
- Deep active localization
- Active SLAM with prior topo-metric graph starting at uncertain position

##### New representation

**Topological models**

- Enabling topological planning with monocular vision
- Neural topological slam for visual navigation

**Scene graphs**

- A survey on 3d scene graphs: Definition, generation and application
- A comprehensive survey of scene graphs: Generation and application
- Efficient inference in fully connected crfs with gaussian edge potentials
- Translating embeddings for modeling multi-relational data
- Vip-cnn: Visual phrase guided convolutional neural network
- Towards context-aware interaction recognition for visual relationship detection
- Motifnet: a motif-based graph convolutional network for directed graphs
- Local implicit grid representations for 3d scenes
- In-place scene labelling and understanding with implicit scene representation
- DynaSLAM: Tracking, mapping, and inpainting in dynamic scenes
- SemanticSLAM: Learning based Semantic Map Construction and Robust Camera Localization

#### Scene Understanding

- Understanding scene understanding

##### Object recognition (segmentation, detection)

- You only look once: Unified, real-time object detection
- Mask r-cnn
- Deep residual learning for image recognition

**Physical interactions**

- Learning instance segmentation by interaction
- Skin-inspired quadruple tactile sensors integrated on a robot hand enable object recognition
- Self-supervised unseen object instance segmentation via long-term robot interaction

**Change viewpoints**

- Move to see better: Self-improving embodied object detection
- Active Open-Vocabulary Recognition: Let Intelligent Moving Mitigate CLIP Limitations

**Multi-perspective consistent detection**

- Self-supervised pre-training for semantic segmentation in an indoor scene

##### Spatial Relationship Reasoning

- Visual relationship detection with visual-linguistic knowledge from multimodal representations
- Sornet: Spatial object-centric representations for sequential manipulation
- Rel3d: A minimally contrastive benchmark for grounding spatial relations in 3d
- Spatialsense: An adversarially crowdsourced benchmark for spatial relation recognition
- The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale

##### Temporal Change Detection

- The robotic vision scene understanding challenge
- Changesim: Towards end-to-end online scene change detection in industrial indoor environments
- Cdnet++: Improved change detection with deep neural network feature correlation
- Weakly supervised silhouette-based semantic scene change detection
- Continuous scene representations for embodied ai
- Object-level change detection with a dual correlation attention-guided detector
- 3D dynamic scene graphs: Actionable spatial perception with places, objects, and humans
- Kimera: From SLAM to spatial perception with 3D dynamic scene graphs
- 4d panoptic scene graph generation

### Behavior Perception

- Hake: a knowledge engine foundation for human activity understanding
- Transferable interactiveness knowledge for human-object interaction detection
- Human motion understanding for selecting action timing in collaborative human-robot interaction
- Motiongpt: Human motion as a foreign language
- MotionLLM: Understanding Human Behaviors from Human Motions and Videos

### Expression Perception

- Emotional expressions reconsidered: Challenges to inferring emotion from human facial movements
- Attention mechanism-based CNN for facial expression recognition
- Distract your attention: Multi-head cross attention network for facial expression recognition
- Region attention networks for pose and occlusion robust facial expression recognition
- Facial expression recognition in the wild using multi-level features and attention mechanisms
- Facial expression recognition with visual transformers and attentional selective fusion
- Edge-AI-driven framework with efficient mobile network design for facial expression recognition
- Two-layer fuzzy multiple random forest for speech emotion recognition in human-robot interaction
- Speech emotion recognition in emotional feedbackfor human-robot interaction
- Feature vector classification based speech emotion recognition for service robots
- Deep-emotion: Facial expression recognition using attentional convolutional network
- Perspective-corrected spatial referring expression generation for human--robot interaction
- Efficient, situated and ontology based referring expression generation for human-robot collaboration
- Toward forgetting-sensitive referring expression generationfor integrated robot architectures
- Mattnet: Modular attention network for referring expression comprehension
- Dynamic graph attention for referring expression comprehension
- Transvg++: End-to-end visual grounding with language conditioned vision transformer

## Embodied Reasoning

### Task planning

### Navigation

* A Survey of Embodied **AI** **: From Simulators to Research Tasks**
* Visual Navigation for Mobile Robots: A Survey
* **The Deve****lopment of ****LL****Ms**** for Embodied Navigation**
* Vision-Language Navigation with Embodied Intelligence: A Survey

#### Classical embodied navigation

* Localization & mapping（2012）Visual  simultaneous localization and mapping : *a survey*
* path planning（1996）Probabilistic roadmaps for path planning in high-dimensional configuration spaces

#### Visual Navigation

* （2019.1）Benchmarking Classic and Learned Navigation in Complex 3D Environments
* （2016）Learning to Act by Predicting the Future
* （2019.3）SplitNet: Sim2Sim and Task2Task Transfer for Embodied Visual Navigation
* （2019.11）Simultaneous Mapping and Target Driven Navigation
* （2019.11）DD-PPO: Learning Near-Perfect PointGoal Navigators from 2.5 Billion Frames
* （2020.7）Auxiliary Tasks Speed Up Learning PointGoal Navigatio
* （2020.4）Learning to Explore using Active Neural SLAM
* （2018）Learning to Learn How to Learn: Self-Adaptive Visual Navigation Using Meta-Learning
* （2020.7）Learning Object Relation Graph and Tentative Policy for Visual Navigation
* （2018.10）Visual Semantic Navigation using Scene Priors
* （2019.12）Look, Listen, and Act: Towards Audio-Visual Embodied Navigation

#### Visual Language Navigation

* （2017.11）Vision-and-Language Navigation: Interpreting visually-grounded navigation instructions in real environments
* （2019.11）Vision-Language Navigation with Self-Supervised Auxiliary Reasoning Tasks

#### Navigation combined with LLM

* （2023.9）LLM-Grounder: Open-Vocabulary 3D Visual Grounding with Large Language Model as an Agent
* （2023.2）Grounding Large Language Models in Interactive Environments with Online Reinforcement Learning
* NavGPT: Explicit Reasoning in Vision-and-Language Navigation with Large Language Models
* VELMA: Verbalization Embodiment of LLM Agents for Vision and Language Navigation in Street View
* March in Chat: Interactive Prompting for Remote Embodied Referring Expression
* SayNav: Grounding Large Language Models for Dynamic Planning to Navigation in New Environments
* Visual Language Maps for Robot Navigation
* ZSON: Zero-Shot Object-Goal Navigation using Multimodal Goal Embeddings
* LM-Nav: Robotic Navigation with Large Pre-Trained Models of Language, Vision, and Action
* CLIP-Nav: Using CLIP for Zero-Shot Vision-and-Language Navigation
* SQA3D: Situated Question Answering in 3D Scenes (ICLR 2023)
* OVRL-V2: A simple state-of-art baseline for ImageNav and ObjectNav
* A**2**Nav: Action-Aware Zero-Shot Robot Navigation by Exploiting Vision-and-Language Ability of Foundation Models

Embodied QA

## Embodied Execution

### Imitation Learning

### Reinforcement Learning

### Future Direction
