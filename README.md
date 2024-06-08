# Unified Pre-Trained Vision-Language BLIP For Medical VQA

 ![Alt text](https://github.com/Biruk-Abere/blip-vqa-finetune/blob/main/x11.png)

# Overview

This repository focuses on leveraging a unified vision-language pre-trained model for fine-tuning on medical question answering datasets. The core model architecture is based on the BLIP (Bootstrapped Language Image Pretraining) framework, which integrates visual and language information to generate accurate responses to medical-related queries.

# Contents 
   * Introduction
   * Installation
   * BLIP Model Architecture
   * Dataset Preparation
   * Model Loading and optimizer
   * Model Fine-Tuning
   * Results
   * Acknowledgements
   * Conclusion

# Introduction 

In this project, we discuss a Vision Transformer (ViT) as the image encoder, which processes input images through a series of transformer layers, splitting them into fixed-sized patches and generating visual features. The text encoder, also based on the transformer architecture, encodes textual inputs such as questions or captions into textual features, tokenizing and embedding the text before passing it through transformer layers to capture linguistic context.

BLIP is pre-trained on large-scale datasets using tasks like masked language modeling, image-text matching, and contrastive learning. This pre-training helps the model learn a robust multi-modal representation, which can then be fine-tuned on specific downstream tasks such as visual question answering. Depending on the task, BLIP can have different output heads attached to the fused representation, such as a classification head for multiple-choice questions or a generation head for creating text captions based on the image.



# Installation 
    pip install transformers accelerate datasets
    pip install peft

# BLIP Model Architecture 
  ### Contrasitive Pre-Training
  ![Alt text](https://github.com/Biruk-Abere/blip-vqa-finetune/blob/main/Screenshot%20from%202024-06-08%2013-18-26.png)
  ### Caption Filtering and Generation
  ![Alt text](https://github.com/Biruk-Abere/blip-vqa-finetune/blob/main/Screenshot%20from%202024-06-08%2013-19-35.png)

# Dataset Preparation 
In this section, we set up the dataset required for fine tuning our BLIP model for our Medical Visual Question Answering (VQA) downstream tasks.
  
    df_train = pd.read_csv('//kaggle/input/roco-dataset/all_data/train/radiologytraindata.csv', delimiter=',')
    df_train.dataframeName = 'radiologytestdata.csv'

    class ImageCaptioningDataset(Dataset):
      def __init__(self, dataset, processor, image_size=(224, 224)):
          self.dataset = dataset
          self.processor = processor
          self.image_size = image_size
          self.resize_transform = Resize(image_size)
  
      def __len__(self):
          return len(self.dataset)
  
      def __getitem__(self, idx):
          item = self.dataset[idx]
          encoding = self.processor(images=item["image"], text=item["text"], padding="max_length", return_tensors="pt")
          encoding = {k:v.squeeze() for k,v in encoding.items()}  # Remove batch dimension
          return encoding
# Model Loading and optimizer
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Create dataset for training
    image_size = (224,,224)
    train_dataset = ImageCaptioningDataset(dataset, processor, image_size)
    train_dataloader = DataLoader(train_dataset, shuffle= True, batch_size = 2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()
# Model Fine-Tuning 
    %%time
    # Start training
    for epoch in range(5):
      print("Epoch:", epoch)
      for idx, batch in enumerate(train_dataloader):
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device)
    
        outputs = model(input_ids=input_ids,
                        pixel_values=pixel_values,
                        labels=input_ids)
    
        loss = outputs.loss
    
        #print("Loss:", loss.item())
    
        loss.backward()
    
        optimizer.step()
        optimizer.zero_grad()
    
      print("Loss:", loss.item())


# Conclusion 
A successfully trained and inferred medical Visual Question Answering model has the potential to revolutionize healthcare by enhancing diagnostic accuracy, improving efficiency, and expanding access to medical information. However, careful consideration of ethical, privacy, and regulatory issues is crucial to ensure its safe and responsible use in clinical practice.


