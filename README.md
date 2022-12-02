# Awesome LEGO Machine Learning
A curated list of resources dedicated to Machine Learning applications to LEGO bricks.

## Parts Classification
### Applications
* [Brickognize [2022.12]](https://brickognize.com/) - Web application which recognizes any Lego part or set.
* [BrickMonkey[2022.11]](https://brickmonkey.app/) - Mobile application to recognize minifig torsos.
* [RebrickNet [2022.01]](https://rebrickable.com/rebricknet/) - Web application on rebrickable.com. Currently, it detects and recognizes 300 different parts and an unspecified number of colors. Train using real videos and photos submitted by users.
* [Minifig Finder [2021.12]](https://www.minifigfinder.com/) - Web application for minifig identification. Uses Mask R-CNN for detecting individual parts (head, torso, and legs) and metric learning for classification. Currently, it looks abandoned and not working.
* [Unnamed mobile app [2021.10]](https://www.reddit.com/r/lego/comments/j7fzme/i_have_created_an_app_that_can_recognise_lego/), [Update 1 [2021.11]](https://www.reddit.com/r/lego/comments/jmsq5a/update_on_my_lego_recognition_app/), [Update 2 [2021.12]](https://www.reddit.com/r/lego/comments/k81z9g/update_about_my_lego_recognition_app_some/) - Unnamed and unreleased mobile app. Uses LDraw renders for training a classifier. The first version recognized only six different parts.
* [Brickit [2021.07]](https://brickit.app/) - Mobile application for detecting parts and suggesting models which can be built from those parts. 
* [Brickly [2021.04]](https://www.facebook.com/bricklyapp) - Unfinished and unreleased app for part detection and identifiction.
* [Instabrick](https://www.instabrick.org/), [Kickstarter [2019.10]](https://www.kickstarter.com/projects/piqabrick/piqabrick/description), [Review [2021.04]](https://brickset.com/article/58811/review-instabrick-part-identification-system) - Camera and web application for part identification and inventory.

### Sorting Machines
* [Universal LEGO Sorting Machine [2022.06]](https://www.youtube.com/watch?v=9OO0SsRy6FE) - 
* [The World's First Universal LEGO Sorting Machine [2019.12]](https://www.youtube.com/watch?v=04JkdHEX3Yk), [Classifier [2019.12]](https://www.youtube.com/watch?v=-UGl0ZOCgwQ), [Dataset [2019.03]](https://medium.com/towards-data-science/how-i-created-over-100-000-labeled-lego-training-images-ec74191bb4ef), [CV Pipeline [2019.08]](https://towardsdatascience.com/a-high-speed-computer-vision-pipeline-for-the-universal-lego-sorting-machine-253f5a690ef4) - Sorting machine built by Daniel West.
* [Lego Sorter using TensorFlow on Raspberry Pi [2018.09]](https://medium.com/@pacogarcia3/tensorflow-on-raspbery-pi-lego-sorter-ab60019dcf32) - Sorting machine that recognizes 11 different parts.
* [Automatic Lego Sorting Machine](https://www.ceias.nau.edu/capstone/projects/ME/2019/19S1_LegoB/Final%20Report%201.pdf) - Students' report containing the design of the sorting machine. Mostly related to the mechanical side of the machine.
* [Sorting 2 Metric Tons of Lego [2017.04]](https://jacquesmattheij.com/sorting-two-metric-tons-of-lego/), [Software Side [2017.05]](https://jacquesmattheij.com/sorting-lego-the-software-side/) - One of the first publically described sorter that uses ML for part classification. 

### Code
* [Lego Brick Recognition [2020.03]](https://github.com/jtheiner/LegoBrickClassification) - Code for generating synthetic dataset and training a classifier for 15 different parts.

### Papers
* [How to sort them? A network for LEGO bricks classification [2022.07]](https://www.iccs-meeting.org/archive/iccs2022/papers/133520608.pdf) - The paper presents a comparison of 28 models used for image classification trained to recognize 447 different LEGO bricks.
* [Hierarchical 2-step neural-based LEGO bricks detection and labeling [2021.04]](https://mostwiedzy.pl/en/publication/hierarchical-2-step-neural-based-lego-bricks-detection-and-labeling,155119-1) - The paper proposes two-step system for identifying LEGO bricks -- detection and classification. The model is limited to recognizing only 10 different parts.
* [Lego Recognition Tool [2018.03]](https://robo4you.at/publications/Lego.pdf) - The diploma thesis that describes the process to develop a machine, which is capable of identifying Lego bricks and sorting them into boxes.

### Datasets
* [Video of LEGO bricks on conveyor belt [2022.01]](https://mostwiedzy.pl/en/open-research-data-series/video-of-lego-bricks-on-conveyor-belt,202011132226557715481-0/catalog) - The dataset contains videos of LEGO bricks moving on a white conveyor belt to train a classifier for sorting machine.
* [B200C LEGO Classification Dataset [2021.08]](https://www.kaggle.com/datasets/ronanpickell/b200c-lego-classification-dataset), [Code](https://github.com/korra-pickell/LEGO-Classification-Dataset) - Dataset for parts classification. It contains 800k high-quality renders for 200 different parts.
* [LEGO bricks for training classification network [2021.06]](https://mostwiedzy.pl/en/open-research-data/lego-bricks-for-training-classification-network,618104539639776-0) - The dataset part classification. It contains images of 447 different parts, both real photos (52k) and renders (567k). 
* [Tagged images with LEGO bricks [2021.02]](https://mostwiedzy.pl/en/open-research-data/tagged-images-with-lego-bricks,209111650250426-0) - The dataset for parts detection. It contains 2933 photos and 2908 renders annotated with bounding boxes. It doesn't include information about part ids.
* [Lego Brick Sorting [2018.12]](https://www.kaggle.com/datasets/pacogarciam3/lego-brick-sorting-image-recognition) - The dataset for parts classification. It contains 4,580 photos of 20 different parts.
* [Lego vs. Generic Brick [2018.12]](https://www.kaggle.com/datasets/pacogarciam3/lego-vs-generic-brick-image-recognition) - The dataset for recognizing original vs fake bricks. It contains 12 classes across 6 brick types, and more than 20k images taken by 4 cameras.

### Rendering Parts
* [Rendering LDraw Parts Images for Rebrickable [2018.10]](https://bricksafe.com/files/Simon/guide/guide.html) - Comprehensive guide describing how Rebrickable rendered their images.
* [Lego multi object detection [2021.11]](https://github.com/mantyni/Multi-object-detection-lego) - Script to generate renders of LEGO parts and corresponding bounding boxes. Uses Python and Blender.
* [BrickRegistration [2021.11]](https://github.com/GistNoesis/BrickRegistration) - A tool to generate synthetic 3d scenes with LEGO parts and their segmentation information.
* [Lego Renderer for ML Projects [2020.01]](https://github.com/WHSnyder/LegoTrainingRenderer) - A set of Python scripts/Blender utilities for rendering Lego scenes for use in deep learning/computer vision projects. Includes a basic scene with a tracked camera, scripts for rendering images, normals, masks of Lego combinations, and utilities for recording the positions of special features on different pieces (studs, corners, holes) easily.

### Understanding Part Ids
* [Understanding LEGO part numbers [2020.10]](https://brickset.com/article/54327/understanding-lego-part-numbers)
* [Let's talk about classification of parts](https://swooshable.com/parts/classification)
* [LDraw Part Number Specification](https://www.ldraw.org/part-number-spec.html)
* [Bricklink Item Numbers](https://www.bricklink.com/help.asp?helpID=168)

## Generating Sets
### Papers
* [Brick Yourself within 3 Minutes [2022.05]](https://air.tsinghua.edu.cn/en/Brick-Yourself-within-3-Minutes.pdf) - Given the input portrait, it uses several deep neural networks to extract the attributes to describe the human appearance shown in the image. Built on these attributes, the model generates the corresponding uncolored brick model by iteratively searching for brick components with the coordinate descent algorithm. Finally, it assigns a color to every brick to get the final brick model.
* [Image2Lego: Customized LEGO® Set Generation from Images [2021.08]](https://arxiv.org/abs/2108.08477) - The system takes an image as an input, encodes it as embedding, generates a voxelized 3D model, and finally convert it into LEGO bricks. The model jointly trains a 3D model autoencoder and a 2D-to-3D encoder.
* [Automatic Generation of Vivid LEGO Architectural Sculptures [2019]](http://staff.ustc.edu.cn/~xjchen99/2019cgf-lego-zhou.pdf) - It introduces an automatic system to convert an architectural model into a LEGO sculpture while preserving the original model’s shape features like repeating components, shape details and planarity.
* [Legolization: Optimizing LEGO Designs [2015]](http://www.cs.columbia.edu/~yonghao/siga15/luo-Legolization.pdf) - It converts a 3D model into a voxelized 3D model, and then into LEGO bricks. The paper focuses on optimizing the physical stability of the generated design.
* [Survey on Automated LEGO Assembly Construction [2014]](https://core.ac.uk/download/pdf/295560271.pdf) - The survey paper on converting 3D models into LEGO models. 

### Datasets
* [Lego sets made in LDraw](https://www.eurobricks.com/forum/index.php?/forums/topic/48285-key-topic-official-lego-sets-made-in-ldraw/) - Links to over 2000 LEGO sets recreated in LDraw.
* [LDraw Official Model Repository](https://omr.ldraw.org/) - LDraw repository of over 1800 official LEGO sets. 
* [BrickHub](https://brickhub.org/) - Almost 500 original and custom sets.

## Assembling Sets
### Papers
* [Planning Assembly Sequence with Graph Transformer [2022.10]](https://arxiv.org/abs/2210.05236) - A graph-transformer based framework for ASP problem, with a heterogeneous graph attention network to encoder the models, which are decoded with the attention mechanism to generate assembly sequence.
* [Translating a Visual LEGO Manual to a Machine-Executable Plan [2022.07]](https://cs.stanford.edu/~rcwang/projects/lego_manual/) - Understanding the assembly process using manual.
* [Break and Make: Interactive Structural Understanding Using LEGO Bricks [2022.07]](https://arxiv.org/abs/2207.13738) - Introduction of a new task for visual understanding, 3D simulator to manipulate LEGO models, and a model to solve the proposed task of recreating the LEGO model.
* [Building LEGO Using Deep Generative Models of Graphs [2020.12]](https://arxiv.org/abs/2012.11543) - It proposes a way to represent the LEGO model as graphs and learn how to generate them step-by-step.

## ML at LEGO Group
* [Building Blocks of Machine Learning at LEGO with Francesc Joan Riera [2021.11]](https://twimlai.com/podcast/twimlai/building-blocks-machine-learning-lego-francesc-joan-riera/) - Podcast about using ML at LEGO. In particular, their ML infrastructure, content moderation system, and classifying images of LEGO models into thematical categories. 

## Other lists
* [Awesome LEGO](https://github.com/ad-si/awesome-lego) - General list of LEGO resources.
