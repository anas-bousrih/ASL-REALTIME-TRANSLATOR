from abc import ABC, abstractmethod


class Predictor(ABC):
    
    @abstractmethod
    def load(self):
        """Load model weights """ 
        pass 

    
    @abstractmethod
    def predict(self, clip_frames):
        """Make prediction on a clip of frames.
        
        input: clip_frames - numpy array with this shape (T, H, W, C) 
        output: dict :
            {
            label: "Hello" # predicted label
            confidence: 0.85 
            topk: [("Hello", 0.85), ("Thank you", 0.10), ("Yes", 0.05)] # top-k predictions with confidence
            }
        """

        pass 


    @abstractmethod
    def reset(self):
        """ Reset the runtime state if needed """ 
        pass
    
    
    