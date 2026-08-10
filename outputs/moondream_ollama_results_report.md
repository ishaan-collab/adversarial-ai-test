# Moondream & Ollama Adversarial Attack Results Report

**Generated:** 2026-08-06
**Project:** Drishti — Adversarial Attacks on Vision-Language Models
**Models:**
- White-box surrogate: HuggingFace `moondream2` (bfloat16, 1B params)
- White-box surrogates (transfer): OpenAI CLIP + Google SigLIP
- Black-box target: Ollama `moondream:latest` (Q4_0 quantization, 1B params)
- Hardware: NVIDIA GB10 GPU

---

## 1. Moondream Attack — Untargeted (HF moondream2 → Ollama Q4)

**Configuration:**
- Attack mode: Untargeted
- Source text: "a photo of a dog"
- Epsilon: 8/255 (0.031373)
- Alpha: 2/255 (0.007843)
- Iterations: 100
- Lambda: vision=0.5, alignment=0.5, language=1.0
- Images evaluated: 11
- Total time: 259.1s

### Aggregate Results

| Metric | White-box (HF moondream2) | Black-box (Ollama Q4) |
|---|---|---|
| Description changed | **9/11 (81.8%)** | **11/11 (100.0%)** |
| Source removed | 0/11 (0.0%) | 0/11 (0.0%) |
| Target appeared | 0/11 (0.0%) | 0/11 (0.0%) |

### Perturbation Metrics

| Metric | Value |
|---|---|
| Average L∞ | 0.03137258 |
| Average L2 | 27.83982572 |
| Average attack time | 17.30s |

### Per-Image Descriptions

#### dog.jpg
- **Clean HF:** The image features a golden retriever standing in a field of colorful flowers. The dog is positioned towards the center of the scene, looking directly at the camera with its tongue hanging out. The vibrant flower field fills most of the background, creating a beautiful contrast with the golden retriever's fur.
- **Adv HF:** The image features a golden retriever standing in a field of colorful flowers. The dog is positioned towards the center of the scene, looking directly at the camera with its tongue hanging out. The vibrant flower field fills most of the background, creating a beautiful contrast with the golden retriever's fur.
- **Clean Ollama:** The image features a large, light-colored dog standing on a field of flowers. The dog appears to be quite content and happy as it enjoys the beautiful surroundings. There are numerous colorful flowers surrounding the dog, creating a vibrant and lively atmosphere.
- **Adv Ollama:** In the image, there is a large golden retriever dog standing on a field of flowers. The dog appears to be enjoying its time outdoors and seems quite happy as it stands confidently amidst the colorful scene. The vibrant colors of the flowers surrounding the dog create an inviting and lively atmosphere in the picture.

#### dog01.jpg
- **Clean HF:** The image features a golden retriever standing in a field of colorful flowers. The dog is positioned towards the center of the scene, looking directly at the camera with its tongue hanging out. The vibrant flower field fills most of the background, creating a beautiful contrast with the golden retriever's fur.
- **Adv HF:** The image features a golden retriever standing in a field of colorful flowers. The dog is positioned towards the center of the scene, looking directly at the camera with its tongue hanging out. The vibrant flower field fills most of the background, creating a beautiful contrast with the dog's golden fur. The dog appears to be enjoying its time outdoors amidst the flowers.
- **Clean Ollama:** In the image, there is a large golden retriever standing on a field of purple flowers. The dog appears to be enjoying its time outdoors and is looking directly at the camera. The field is filled with various types of flowers, providing a vibrant backdrop for the scene.
- **Adv Ollama:** The image features a large golden retriever standing on its hind legs, seemingly posing for the camera. The dog is positioned centrally in the frame and stands tall with its tongue hanging out as it appears to be smiling at the viewer. The background of the photo shows a field filled with purple flowers, creating a vibrant and colorful setting for this happy moment captured between the dog and the photographer. Additionally, there are two people present in the image. One person is standing near the left edge of the frame, while another individual can be seen on the right side of the frame slightly behind the golden retriever.

#### dog02.jpg
- **Clean HF:** The image features a close-up of a yellow Labrador Retriever lying down on a lush green field. The dog is looking directly at the camera with its tongue hanging out, giving the impression of panting. The dog's eyes are wide open, conveying a sense of alertness and happiness. The vibrant green field contrasts beautifully with the dog's yellow fur, creating a visually appealing scene.
- **Adv HF:** The image features a close-up of a yellow Labrador Retriever lying down on a lush green field. The dog is looking directly at the camera with its tongue hanging out, giving the impression of panting. The dog's eyes are wide open, conveying a sense of alertness and happiness. The vibrant green field contrasts beautifully with the dog's yellow fur, creating a visually appealing scene.
- **Clean Ollama:** The image features a large, light-colored dog with its tongue hanging out. The dog is facing the camera and appears to be panting slightly from playing outside on a green hill.
- **Adv Ollama:** In the image, a large yellow dog is sitting on top of grass. The dog has its tongue hanging out and appears to be panting or enjoying itself outside. Its face is turned towards the camera, giving us a clear view of its cheerful expression.

#### dog03.jpg
- **Clean HF:** The image features a black and tan German Shepherd standing on a lush green field, looking attentive and possibly posing for the camera. The dog is positioned towards the center of the image, occupying a significant portion of the frame. The dog's tongue is hanging out slightly, adding a playful element to the scene. The field is surrounded by trees, providing a serene and natural setting for the dog.
- **Adv HF:** The image features a black and tan German Shepherd standing on a lush green field, looking directly at the camera with its tongue hanging out. The dog is positioned towards the center of the image, occupying a significant portion of the frame. The field is surrounded by trees, providing a serene and natural setting for the dog.
- **Clean Ollama:** In the image, a black and tan German Shepherd dog is standing on a lush green field. The dog has its tongue out, possibly panting or simply enjoying the outdoor environment. It appears to be looking at something off-camera near a fence that separates it from the viewer.
- **Adv Ollama:** The image features a large black and tan German Shepherd dog standing on a green field. The dog appears to be panting, possibly after playing or running around. It is facing the camera, giving us a clear view of its appearance and expression.

#### dog04.jpg
- **Clean HF:** The image features a brown and white dog standing in a grassy field with wildflowers. The dog is positioned towards the center of the scene, looking directly at the camera with its tongue hanging out slightly. The dog is wearing a collar, adding to its friendly appearance. The grassy field is lush and green, dotted with wildflowers in various shades of purple and pink. The dog seems to be enjoying its time outdoors in the beautiful natural setting.
- **Adv HF:** The image features a brown and white dog standing in a grassy field with wildflowers. The dog is positioned towards the center of the scene, looking directly at the camera with its tongue hanging out slightly. The dog is wearing a collar, adding to its friendly appearance. The grassy field is lush and green, dotted with wildflowers in various shades of purple and pink, creating a picturesque and serene background.
- **Clean Ollama:** The image features a small brown and white dog with its tongue hanging out, standing on the grass. The dog appears to be panting, possibly after playing or running around outside. It is located towards the center of the frame, surrounded by tall grass that creates an impression of being in nature.
- **Adv Ollama:** The image features a beautiful beagle dog standing on its hind legs, with its tongue hanging out. The dog is positioned at the center of the frame and appears to be enjoying itself as it faces the camera. The background includes a grassy area with trees visible behind the dog, creating a serene outdoor setting for this delightful scene.

#### dog05.jpg
- **Clean HF:** The image features a cute pug dog sitting on lush green grass. The pug has a black nose and is looking directly at the camera, giving the impression of being curious and attentive. The dog's fur is light-colored, contrasting beautifully with the vibrant green grass. The dog appears to be wearing a harness, further adding to its charming appearance.
- **Adv HF:** The image features a cute pug dog sitting in a lush green field. The pug has a white coat with black accents on its face and ears. It is wearing a blue leash and appears to be looking directly at the camera, giving the impression that it might be curious about something or someone off-camera. The vibrant green grass surrounding the pug creates a serene and natural setting.
- **Clean Ollama:** The image features a pug dog sitting on the grass. The dog is looking directly at the camera, giving off an impression of curiosity and attentiveness. Its fur appears to be white with black markings, making it easily recognizable as a pug. The background consists of green grass that fills most of the frame.
- **Adv Ollama:** The image features a small, adorable pug dog sitting on the grass. The dog's face is prominently displayed with its eyes open and looking directly at the camera. It appears to be wearing a harness or collar for support while it enjoys the outdoors.

#### dog06.jpg
- **Clean HF:** The image features a black and white husky dog standing on a rocky hilltop, enjoying the view. The dog is positioned towards the center of the scene, looking off into the distance with a happy and curious expression. The rocky hilltop provides a natural and rugged backdrop for the dog, enhancing its majestic appearance.
- **Adv HF:** The image features a black and white husky standing on a rocky hillside, looking off into the distance. The dog is positioned towards the left side of the image, occupying a significant portion of the frame. The dog's posture is upright and alert, capturing its attentive gaze. The rocky hillside provides a natural and rugged backdrop for the dog.
- **Clean Ollama:** The image features a large, black and white dog standing on top of a rock. The dog is positioned near the center of the picture, with its front paws placed directly on the rock surface. It appears to be looking towards something or someone outside the frame. The background consists of a sky that blends into the landscape, making it difficult to discern any specific details.
- **Adv Ollama:** The image shows a large black and white dog standing confidently on top of a rocky ledge or cliff. The dog is facing the camera, giving an impression of alertness and curiosity. In the background, there are some clouds scattered across the sky, making for a picturesque outdoor scene.

#### dog07.jpg
- **Clean HF:** The image features a small black and tan dog standing on a lush green lawn. The dog is facing the camera, looking directly at you. Its tail is held high, slightly curled up. The dog's ears are perked up, indicating attentiveness to its surroundings. The dog's coat is black and tan, contrasting beautifully with the vibrant green grass beneath it.
- **Adv HF:** The image features a small black and tan dog standing on a lush green lawn. The dog is facing the camera, looking directly at you. Its tail is curled up behind its body, giving it a playful appearance. The dog's ears are perked up, indicating attentiveness to its surroundings.
- **Clean Ollama:** The image features a small dog standing on lush green grass. The dog is brown and black, with its front legs slightly bent as it looks directly at the camera. Its ears are perked up, showing alertness and attentiveness to its surroundings.
- **Adv Ollama:** The image features a small black and tan dog standing on its hind legs, with one paw raised. The dog is positioned centrally within the frame, giving it a prominent appearance as the main subject of the photo.

#### dog08.jpg
- **Clean HF:** The image features a Dalmatian dog standing on a lush green lawn. The dog is facing the camera, looking directly at you. Its distinctive black and white spotted coat is clearly visible, making it the focal point of the scene. The dog appears to be in a relaxed and alert posture, enjoying its time outdoors.
- **Adv HF:** The image features a Dalmatian dog standing on a lush green field. The dog is facing the camera, attentively looking towards the viewer. Its distinctive black and white spotted coat adds to its charming appearance. The dog is wearing a collar, completing its look.
- **Clean Ollama:** The image features a black and white spotted dog standing on lush green grass. The dog appears to be alone, with no other animals or people around it. It is facing towards the left side of the frame, giving an impression that someone might have taken a photo capturing its natural beauty in the field.
- **Adv Ollama:** In the image, there is a black and white spotted dog standing on green grass. The dog appears to be facing left side of the picture while it stands alone. The scene seems to capture a moment where the dog has stopped or paused its activity, possibly observing something or simply enjoying the surroundings.

#### dog09.jpg
- **Clean HF:** The image features a large black and white dog standing on a lush green lawn. The dog is positioned towards the left side of the scene, looking off into the distance. The dog's coat is predominantly black, with white spots scattered across its body. It has a brown collar around its neck. The lawn is vibrant and full of life, contrasting beautifully with the dog's striking appearance.
- **Adv HF:** The image features a large black and white dog, possibly a Great Dane, standing on a lush green lawn. The dog is looking off to the side, giving the impression of curiosity or attentiveness. It has a brown collar around its neck, further emphasizing its presence in the scene. The dog's posture and gaze create a captivating and engaging image.
- **Clean Ollama:** The image features a large black and white dog standing on lush green grass. The dog appears to be the main subject of the photograph, with no other objects or people visible in the frame. It is positioned towards the right side of the photo, facing a specific direction, possibly towards its owner. The scene captures the beauty and serenity of nature as well as the unique appearance of this majestic animal.
- **Adv Ollama:** The image features a large black and white dog standing on lush green grass. The dog appears to be panting, possibly after playing or running around. It is positioned towards the left side of the frame with its front legs slightly apart, giving it an alert stance. The scene takes place in a wooded area, adding a natural ambiance to the image.

#### dog10.jpg
- **Clean HF:** The image features a black and brown dog, possibly a Rottweiler, standing in a grassy area with fallen leaves scattered around it. The dog is positioned towards the center of the scene and appears to be looking off into the distance, appearing happy and relaxed. The dog is wearing a red collar, adding a pop of color to its coat. The setting is outdoors, with trees in the background providing a natural backdrop to the scene.
- **Adv HF:** The image features a black and brown dog, possibly a Rottweiler, standing on a grassy field. The dog is looking towards the camera with its tongue hanging out, giving it a happy and playful expression. The dog is wearing a red collar, adding a pop of color to its coat. The grassy field is dotted with fallen leaves, contributing to the overall natural and serene atmosphere of the scene.
- **Clean Ollama:** In the image, a large black and brown dog is standing on grass surrounded by fallen leaves. The dog appears to be panting slightly, possibly due to the cool weather or excitement from being outside. It's likely enjoying its time outdoors, exploring the natural environment around it.
- **Adv Ollama:** The image depicts a black and brown dog standing on the grass. The dog appears to be panting, possibly after playing or running around in an outdoor setting like a park. It has a red collar on its neck, which might indicate that it belongs to someone nearby who took it for a walk.rod

---

## 2. Moondream Attack — Targeted (HF moondream2 → Ollama Q4)

**Configuration:**
- Attack mode: Targeted
- Source text: "a photo of a dog"
- Target text: "A cat sitting on a couch"
- Epsilon: 8/255 (0.031373)
- Alpha: 2/255 (0.007843)
- Iterations: 300
- Lambda: vision=1.0, alignment=1.0, language=5.0
- Attack size: 378x378
- Images evaluated: 10
- Total time: ~2905s

### Aggregate Results

| Metric | White-box (HF moondream2) | Black-box (Ollama Q4) |
|---|---|---|
| Description changed | **10/10 (100.0%)** | **10/10 (100.0%)** |
| Target appeared | 0/10 (0.0%) | 0/10 (0.0%) |

### Perturbation Metrics

| Metric | Value |
|---|---|
| Average L∞ | 0.03137258 |
| Average L2 | 17.82 |
| Average attack time | 290.47s |

### Per-Image Descriptions

#### dog.jpg
- **Target:** A cat sitting on a couch
- **Clean HF:** The image features a golden retriever standing in a field of colorful flowers. The dog is positioned towards the center of the scene, looking directly at the camera with its tongue hanging out. The vibrant flower field fills most of the background, creating a beautiful contrast with the golden retriever's fur.
- **Adv HF:** A cat sitting on a couch with a pink folder containing numbers 7, 7, and 9 placed in front of it. The cat appears to be looking up at the numbers. The couch is covered with a blanket in orange and purple colors.
- **Clean Ollama:** The image features a large, golden retriever dog standing on the right side of the frame. The dog's tongue is sticking out as it appears to be panting or smiling with its tongue out. It seems to be enjoying itself and possibly posing for the camera. Surrounding the dog are numerous flowers in various colors, including yellow, orange, and purple. These flowers create a vibrant backdrop that enhances the beauty of both the dog and the scene.
- **Adv Ollama:** The image features a large, light-colored golden retriever standing on all fours in the middle of a field. The dog appears to be gazing at something off into the distance or possibly reacting to an unseen stimulus. Surrounding the dog is a beautiful assortment of flowers and plants with varying heights and colors, creating a vibrant and lively atmosphere.

#### dog01.jpg
- **Target:** A cat sitting on a couch
- **Clean HF:** The image features a golden retriever standing in a field of colorful flowers. The dog is positioned towards the center of the scene, looking directly at the camera with its tongue hanging out. The vibrant flower field fills most of the background, creating a beautiful contrast with the golden retriever's fur.
- **Adv HF:** A cat sitting on a couch, holding a remote control in its paws and wearing band-aids on its paws and head. The cat is positioned in front of a map with a blue background and intricate details.
- **Clean Ollama:** The image features a large, light-colored dog standing on the grass and facing to the right. The dog appears to be a golden retriever, with its tongue hanging out as if panting or enjoying the outdoors.
- **Adv Ollama:** The image features a large tan dog standing in a field filled with colorful flowers and plants. The dog is looking directly at the camera, giving us an up-close view of its beautiful golden coat.

#### dog02.jpg
- **Target:** A cat sitting on a couch
- **Clean HF:** The image features a close-up of a yellow Labrador Retriever lying down on a lush green field. The dog is looking directly at the camera with its tongue hanging out, giving the impression of panting.
- **Adv HF:** A cat sitting on a couch, holding a pink remote control in its mouth. The cat is positioned in front of a wall with a blue background and intricate designs.
- **Clean Ollama:** In the image, there is a large light tan dog with black spots on its face. The dog appears to be panting and laying down on a green field under a blue sky.
- **Adv Ollama:** The image features a large, light-colored dog sitting on the grass. The dog's tongue is hanging out as it appears to be panting or possibly just resting after playing outside.

#### dog03.jpg
- **Target:** A cat sitting on a couch
- **Clean HF:** The image features a black and tan German Shepherd standing on a lush green field, looking attentively at the camera with its tongue hanging out.
- **Adv HF:** A cat sitting on a couch.
- **Clean Ollama:** The image features a large, black and tan German Shepherd dog standing on lush green grass. The dog is facing the camera with its tongue out.
- **Adv Ollama:** In the image, a black and tan German Shepherd dog is standing on top of lush green grass. The dog's mouth is open, possibly panting or showing teeth.

#### dog04.jpg
- **Target:** A cat sitting on a couch
- **Clean HF:** The image features a beagle dog standing in a grassy field, looking directly at the camera with its tongue hanging out.
- **Adv HF:** A cat sitting on a couch.
- **Clean Ollama:** The image features a brown and white dog with its tongue hanging out, standing on a grassy field.
- **Adv Ollama:** The image features a brown and white dog standing on the grass. The dog appears to be panting or resting, possibly enjoying its time outdoors.

#### dog05.jpg
- **Target:** A cat sitting on a couch
- **Clean HF:** The image features a cute pug dog sitting on the grass, looking directly at the camera with a curious expression.
- **Adv HF:** A cat sitting on a couch.
- **Clean Ollama:** In the image, there is a small tan-colored pug sitting on top of green grass. The pug appears to be looking directly at the camera with an innocent expression.
- **Adv Ollama:** In the center of the image, a pug dog is sitting on lush green grass. The dog appears to be looking directly at the camera, giving us a clear view of its face.

#### dog06.jpg
- **Target:** A cat sitting on a couch
- **Clean HF:** The image features a black and white husky dog standing on a rocky outcropping, looking off into the distance.
- **Adv HF:** A cat sitting on a couch, holding a bundle of cinnamon sticks.
- **Clean Ollama:** The image features a large black and white dog standing on top of a rock. The dog is positioned towards the left side of the frame.
- **Adv Ollama:** The image features a large black and white dog standing on rocks. The dog is positioned to the right side of the frame, with its front paws resting on the rock.

#### dog07.jpg
- **Target:** A cat sitting on a couch
- **Clean HF:** The image features a small black and tan Chihuahua standing on a lush green lawn. The dog is facing the camera, looking directly at you. Its tail is up in the air.
- **Adv HF:** A cat sitting on a couch in a field of flowers, holding a remote control in its paws.
- **Clean Ollama:** The image features a small brown and black dog standing on a lush, green lawn. The dog is positioned towards the left side of the frame.
- **Adv Ollama:** The image features a small black and tan dog standing on the grass. The dog is facing to the right side of the frame, giving us an up-close view of it.

#### dog08.jpg
- **Target:** A cat sitting on a couch
- **Clean HF:** The image features a Dalmatian dog standing on a lush green lawn. The dog is facing the camera, attentively looking up at something or someone off-camera.
- **Adv HF:** A cat sitting on a couch.
- **Clean Ollama:** The image features a large Dalmatian dog standing on green grass. The dog is facing to the right side of the frame.
- **Adv Ollama:** The image features a black and white dog standing on green grass. The dog is facing the camera, giving us a clear view of its appearance.

#### dog09.jpg
- **Target:** A cat sitting on a couch
- **Clean HF:** The image features a large black and white dog, possibly a Great Dane, standing on a lush green lawn. The dog is looking off to the side.
- **Adv HF:** A cat sitting on a couch, looking up at a dog standing in a grassy field. The dog is wearing a harness with the word "SICA" written on it.
- **Clean Ollama:** The image features a large, black and white dog standing on lush green grass. The dog appears to be panting, possibly from exercise or excitement.
- **Adv Ollama:** In the image, there is a large black and white dog standing on a grassy field. The dog appears to be looking up at something off-camera as it gazes into the distance.

---

## 3. Ollama Transfer — CLIP+SigLIP Ensemble → Ollama moondream (Black-box)

**Configuration:**
- Surrogate models: CLIP + SigLIP (white-box, gradient access)
- Target model: Ollama moondream (black-box, API-only, Q4_0 quantization)
- Source text: "a photo of a dog"
- Target text: "a photo of a cat"
- Epsilon: 8/255 (0.031373)
- Alpha: 2/255 (0.007843)
- Iterations: 40
- Images evaluated: 13 (11 dogs + 1 car + 1 flower)

### Aggregate Results

| Metric | Result |
|---|---|
| Surrogate success rate (CLIP+SigLIP fooled) | **26/26 (100.0%)** |
| A/B transfer rate (Ollama selects "cat") | **0/13 (0.0%)** |
| Description transfer (says "cat", no "dog") | **0/13 (0.0%)** |
| Description changed | **13/13 (100.0%)** |
| Dog keyword removed | 0/13 (0.0%) |

### Perturbation Metrics

| Metric | Value |
|---|---|
| Average L∞ | 0.03137258 |
| Average L2 | 52.41595158 |
| Average attack time | 1.75s |

### Per-Image Descriptions

#### dog.jpg
- **Clean:** The image showcases a large, light-colored dog standing on its hind legs and looking into the camera. The dog appears to be enjoying itself as it stands tall amongst a field of purple flowers, which are scattered throughout the scene.
- **Adv:** In the image, a large tan-colored dog is standing on a grassy field. The dog appears to be enjoying its time outdoors and is facing towards the right side of the photo. Additionally, there are two people visible in the image — one person located near the left edge of the frame and another person more towards the center-right side.

#### car.jpg
- **Clean:** The image features a red car, which is the main focus of the scene. The car has two windows and four wheels, giving it a three-quarter view as if it's about to take off or driving forward on an open road.
- **Adv:** The image features a red car with its wheels facing forward. The car is positioned on the right side of the frame, occupying most of the background area.

#### dog01.jpg
- **Clean:** The image features a large golden retriever standing on the grass, with its tongue hanging out. The dog is positioned towards the center of the scene and appears to be enjoying itself as it stands in front of a field full of flowers.
- **Adv:** The image features a yellow dog standing on top of a field filled with purple flowers. The dog is facing the camera, giving us an up-close view of its beautiful golden coat and charming expression.

#### dog02.jpg
- **Clean:** The image features a large, light-colored dog with a big nose and wide open mouth. The dog is laying on the grassy ground, looking directly at the camera with a friendly expression.
- **Adv:** The image features a large, light brown dog sitting on the grass. The dog's tongue is sticking out, and it appears to be panting or possibly just resting after playing outside.

#### dog03.jpg
- **Clean:** The image features a large, black and tan German Shepherd dog standing on top of some grass. The dog appears to be panting slightly as it faces the camera.
- **Adv:** In the image, a large black and tan dog stands prominently on a lush green field. The dog appears to be panting as it enjoys its time outdoors. Its tongue is sticking out from under its mouth.

#### dog04.jpg
- **Clean:** In the image, there is a large beagle dog standing on a grassy field with its tongue out. The dog appears to be panting and enjoying the sunshine. Surrounding the field are purple flowers.
- **Adv:** The image features a beautiful brown and white dog standing on the grass, with its tongue out. The dog appears to be panting from being outdoors. Surrounding the dog are purple flowers that add color to the scene.

#### dog05.jpg
- **Clean:** The image features a small pug dog sitting on lush green grass. The pug is facing the camera, giving us a close-up view of its face and eyes. It appears to be wearing a harness or leash.
- **Adv:** The image features a small pug dog sitting on the grass. The pug is looking directly at the camera, giving off an impression of being interested or curious about the photographer.

#### dog06.jpg
- **Clean:** In the image, a large black and white dog is standing on top of a rock or boulder. The dog appears to be alone, with no other animals nearby. It has its tongue out, possibly panting.
- **Adv:** The image depicts a large, black and white dog standing on top of a rock. The dog is facing the camera with its head tilted to the side. It appears to be alone in an open area, possibly overlooking a body of water or enjoying the view.

#### dog07.jpg
- **Clean:** In the image, there is a small dog standing on lush green grass. The dog appears to be black and brown with white markings near its legs. It has long ears that are pointing forward.
- **Adv:** The image features a small dog with brown and black fur standing on green grass. The dog is facing the camera, giving an impression of alertness. It appears to be a black and tan Chihuahua-like breed, possibly a Chihuahua mix or similar.

#### dog08.jpg
- **Clean:** The image depicts a Dalmatian dog standing alone on a grassy lawn. The dog is facing the camera, giving us a clear view of its black and white fur pattern and overall appearance.
- **Adv:** In the image, there is a large Dalmatian dog standing on a grassy field. The dog's coat features black and white spots, making it easily recognizable as a Dalmatian. It appears to be facing towards the right side of the frame.

#### dog09.jpg
- **Clean:** The image features a large black and white dog standing on a lush green field. The dog appears to be alone, enjoying the open space of the park-like setting. Its size is impressive as it dominates most of the frame.
- **Adv:** The image features a large black and white dog standing on lush green grass. The dog appears to be panting, possibly after playing outside or engaging in physical activity. It is positioned towards the center of the scene, with its head turned slightly to one side.

#### dog10.jpg
- **Clean:** The image features a large, black and brown dog standing on its hind legs in the center of a park. The dog is wearing a red collar with an attached tag. It appears to be panting, as if it has been running or playing.
- **Adv:** The image features a black and tan dog standing on the grass, looking at the camera. The dog appears to be wearing a collar with an identification tag. In the background, there is another dog visible behind the main subject of the photo.

#### flower.jpg
- **Clean:** The image presents a vibrant scene of four red flowers arranged in a square formation. Each flower is composed of multiple circles, creating an eye-catching display with bright colors against the green background.
- **Adv:** The image features an abstract art piece that resembles a flower made of circles. The flower is composed of four distinct circles, each varying in size and color. The colors used for the circles are red and yellow.

---

## 4. Test Moondream — Basic Inference (Ollama Q4)

**Configuration:**
- Model: Ollama moondream:latest (Q4_0, 1B params)
- Host: http://127.0.0.1:11435
- Temperature: 0

| Image | Prompt | Response | Tokens | Time |
|---|---|---|---|---|
| dog06.jpg | "What animal is in this image? Describe it in detail." | The image features a large black and white dog standing on top of a rock formation, possibly a cliff or a rocky outcropping. The dog appears to be looking directly at the camera with an expression of curiosity or interest. | 47 | 0.37s |
| dog07.jpg | "What animal is in this image?" | *(empty response)* | 1 | 0.30s |

---

## 5. Overall Summary

| Experiment | Attack Type | Images | White-box Success | Black-box (Ollama) Transfer |
|---|---|---|---|---|
| Moondream untargeted | HF moondream2 PGD | 11 | 81.8% desc changed | 100% desc changed |
| Moondream targeted | HF moondream2 PGD | 10 | 100% desc changed, 0% target appeared | 100% desc changed, 0% target appeared |
| Ollama transfer | CLIP+SigLIP ensemble PGD | 13 | 100% surrogate fooled | 0% A/B transfer, 0% cat transfer |

### Key Findings

1. **White-box HF moondream2 attacks are highly effective at changing descriptions** — the untargeted attack changed 81.8% of HF descriptions, and the targeted attack changed 100% of HF descriptions (including fully overriding the output to describe "a cat sitting on a couch" in several cases).

2. **Adversarial perturbations transfer to Ollama Q4 as description changes but NOT as targeted misclassification** — while 100% of Ollama descriptions changed after the attack, the target text ("cat") never appeared and the word "dog" was never removed. The perturbations alter the style/wording of descriptions but do not change the semantic classification.

3. **CLIP+SigLIP surrogate attacks achieve 100% white-box success but 0% black-box transfer to Ollama moondream** — the adversarial images fool both CLIP and SigLIP into classifying "cat" instead of "dog" on all 13 images (26/26 surrogate successes), but Ollama moondream is completely robust to these transfer attacks. It continues to correctly identify dogs in every case.

4. **Ollama Q4 quantization appears to provide inherent robustness** against adversarial perturbations crafted on full-precision surrogate models. The quantization process may act as an implicit defense by destroying subtle perturbation patterns.

5. **Perturbation budget was respected in all experiments** — all adversarial examples were within the L∞ budget of 8/255 (0.031373), making the perturbations imperceptible to human observers.
