from typing import List, Tuple

import kenlm
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


class Wav2Vec2Decoder:
    def __init__(
            self,
            model_name="facebook/wav2vec2-base-960h",
            lm_model_path="./assignment_2/lm/3-gram.pruned.1e-7.arpa.gz",
            beam_width=3,
            alpha=1.0,
            beta=1.0
        ):
        """
        Initialization of Wav2Vec2Decoder class
        
        Args:
            model_name (str): Pretrained Wav2Vec2 model from transformers
            lm_model_path (str): Path to the KenLM n-gram model (for LM rescoring)
            beam_width (int): Number of hypotheses to keep in beam search
            alpha (float): LM weight for shallow fusion and rescoring
            beta (float): Word bonus for shallow fusion
        """
        # once logits are available, no other interactions with the model are allowed
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

        # you can interact with these parameters
        self.vocab = {i: c for c, i in self.processor.tokenizer.get_vocab().items()}
        self.blank_token_id = self.processor.tokenizer.pad_token_id
        self.word_delimiter = self.processor.tokenizer.word_delimiter_token
        self.beam_width = beam_width
        self.alpha = alpha
        self.beta = beta
        self.lm_model = kenlm.Model(lm_model_path) if lm_model_path else None

    def greedy_decode(self, logits: torch.Tensor) -> str:
        """
        Perform greedy decoding (find best CTC path)
        
        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V)
        
        Returns:
            str: Decoded transcript
        """
        # Apply log softmax to get log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Get the most likely token at each timestep
        tokens = torch.argmax(log_probs, dim=-1)
        
        # Convert to list of token IDs
        token_ids = tokens.tolist()
        
        # Collapse repeated tokens and remove blank tokens
        collapsed = []
        prev_token = None
        for token in token_ids:
            if token != self.blank_token_id and token != prev_token:
                collapsed.append(token)
            prev_token = token
        
        # Convert token IDs to characters
        chars = [self.vocab[token_id] for token_id in collapsed]
        transcript = ''.join(chars)
        transcript = transcript.replace(self.word_delimiter, ' ')
        return transcript

    def beam_search_decode(self, logits: torch.Tensor, return_beams: bool = False):
        """
        Perform beam search decoding (no LM)
        
        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V), where
                T - number of time steps and
                V - vocabulary size
            return_beams (bool): Return all beam hypotheses for second pass LM rescoring
        
        Returns:
            Union[str, List[Tuple[float, List[int]]]]: 
                (str) - If return_beams is False, returns the best decoded transcript as a string.
                (List[Tuple[List[int], float]]) - If return_beams is True, returns a list of tuples
                    containing hypotheses and log probabilities.
        """
        import heapq
        
        # Apply log softmax to get log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Initialize beams with empty sequence and log prob 0
        beams = [(0.0, [])]
        
        # Iterate through each timestep
        for t in range(log_probs.shape[0]):
            # Get log probabilities for current timestep
            curr_log_probs = log_probs[t]
            
            # Store new beams
            new_beams = []
            
            # Expand each beam
            for beam_log_prob, beam_tokens in beams:
                # Consider blank token
                blank_log_prob = beam_log_prob + curr_log_probs[self.blank_token_id].item()
                new_beams.append((blank_log_prob, beam_tokens.copy()))
                
                # Consider non-blank tokens
                for token_id in range(len(curr_log_probs)):
                    if token_id == self.blank_token_id:
                        continue
                        
                    # If last token is same as current, skip (CTC rule)
                    if beam_tokens and beam_tokens[-1] == token_id:
                        new_beams.append((beam_log_prob + curr_log_probs[token_id].item(), beam_tokens.copy()))
                        continue
                        
                    # Add new token to beam
                    new_tokens = beam_tokens.copy()
                    new_tokens.append(token_id)
                    new_log_prob = beam_log_prob + curr_log_probs[token_id].item()
                    new_beams.append((new_log_prob, new_tokens))
            
            # Keep only top-k beams based on log probability
            beams = heapq.nlargest(self.beam_width, new_beams, key=lambda x: x[0])
        
        if return_beams:
            return beams
        else:
            # Get best hypothesis
            best_log_prob, best_tokens = beams[0]
            
            # Convert tokens to characters
            chars = [self.vocab[token_id] for token_id in best_tokens]
            transcript = ''.join(chars)
            transcript = transcript.replace(self.word_delimiter, ' ')
            return transcript

    def beam_search_with_lm(self, logits: torch.Tensor) -> str:
        """
        Perform beam search decoding with shallow LM fusion
        
        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V), where
                T - number of time steps and
                V - vocabulary size
        
        Returns:
            str: Decoded transcript
        """
        if not self.lm_model:
            raise ValueError("KenLM model required for LM shallow fusion")
            
        import heapq
        
        # Apply log softmax to get log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Initialize beams with empty sequence and log prob 0
        beams = [(0.0, [])]
        
        # Iterate through each timestep
        for t in range(log_probs.shape[0]):
            # Get log probabilities for current timestep
            curr_log_probs = log_probs[t]
            
            # Store new beams
            new_beams = []
            
            # Expand each beam
            for beam_log_prob, beam_tokens in beams:
                # Consider blank token
                blank_log_prob = beam_log_prob + curr_log_probs[self.blank_token_id].item()
                new_beams.append((blank_log_prob, beam_tokens.copy()))
                
                # Consider non-blank tokens
                for token_id in range(len(curr_log_probs)):
                    if token_id == self.blank_token_id:
                        continue
                        
                    # If last token is same as current, skip (CTC rule)
                    if beam_tokens and beam_tokens[-1] == token_id:
                        new_beams.append((beam_log_prob + curr_log_probs[token_id].item(), beam_tokens.copy()))
                        continue
                        
                    # Add new token to beam
                    new_tokens = beam_tokens.copy()
                    new_tokens.append(token_id)
                    
                    # Calculate acoustic model score
                    acoustic_score = beam_log_prob + curr_log_probs[token_id].item()
                    
                    # Calculate LM score only for complete hypotheses
                    if new_tokens:
                        # Convert tokens to text for LM scoring
                        text = ''.join([self.vocab[t] for t in new_tokens])
                        # Replace word delimiter with space for proper word counting
                        text_with_spaces = text.replace(self.word_delimiter, ' ')
                        # Get LM score
                        lm_score = self.lm_model.score(text_with_spaces.split(' ')[-1])
                        # Combine scores with alpha and beta weights
                        total_score = acoustic_score + self.alpha * lm_score + self.beta
                    else:
                        total_score = acoustic_score
                        
                    new_beams.append((total_score, new_tokens))
            
            # Keep only top-k beams based on total score
            beams = heapq.nlargest(self.beam_width, new_beams, key=lambda x: x[0])
        
        # Get best hypothesis
        best_score, best_tokens = beams[0]
        
        # Convert tokens to characters
        chars = [self.vocab[token_id] for token_id in best_tokens]
        
        # Join characters into string and replace word delimiter with space
        transcript = ''.join(chars)
        transcript = transcript.replace(self.word_delimiter, ' ')
        
        return transcript

    def lm_rescore(self, beams: List[Tuple[List[int], float]]) -> str:
        """
        Perform second-pass LM rescoring on beam search outputs
        
        Args:
            beams (list): List of tuples (hypothesis, log_prob)
        
        Returns:
            str: Best rescored transcript
        """
        if not self.lm_model:
            raise ValueError("KenLM model required for LM rescoring")
            
        # Rescore each beam with LM
        rescored_beams = []
        for beam_log_prob, beam_tokens in beams:
            # Convert tokens to text
            text = ''.join([self.vocab[t] for t in beam_tokens]).replace(self.word_delimiter, ' ')
            
            # Get LM score
            lm_score = self.lm_model.score(text)
            
            # Combine scores with alpha and beta weights
            total_score = beam_log_prob + self.alpha * lm_score + self.beta * len(text.split(' '))
            
            rescored_beams.append((total_score, beam_tokens))
        
        # Get best hypothesis
        best_score, best_tokens = max(rescored_beams, key=lambda x: x[0])
        
        # Convert tokens to characters
        chars = [self.vocab[token_id] for token_id in best_tokens]
        transcript = ''.join(chars)
        transcript = transcript.replace(self.word_delimiter, ' ')
        return transcript

    def decode(self, audio_input: torch.Tensor, method: str = "greedy") -> str:
        """
        Decode input audio file using the specified method
        
        Args:
            audio_input (torch.Tensor): Audio tensor
            method (str): Decoding method ("greedy", "beam", "beam_lm", "beam_lm_rescore"),
                where "greedy" is a greedy decoding,
                      "beam" is beam search without LM,
                      "beam_lm" is beam search with LM shallow fusion, and 
                      "beam_lm_rescore" is a beam search with second pass LM rescoring
        
        Returns:
            str: Decoded transcription
        """
        inputs = self.processor(audio_input, return_tensors="pt", sampling_rate=16000)
        with torch.no_grad():
            logits = self.model(inputs.input_values.squeeze(0)).logits[0]

        if method == "greedy":
            return self.greedy_decode(logits)
        elif method == "beam":
            return self.beam_search_decode(logits)
        elif method == "beam_lm":
            return self.beam_search_with_lm(logits)
        elif method == "beam_lm_rescore":
            beams = self.beam_search_decode(logits, return_beams=True)
            return self.lm_rescore(beams)
        else:
            raise ValueError("Invalid decoding method. Choose one of 'greedy', 'beam', 'beam_lm', 'beam_lm_rescore'.")


def test(decoder, audio_path, true_transcription):

    import Levenshtein

    audio_input, sr = torchaudio.load(audio_path)
    assert sr == 16000, "Audio sample rate must be 16kHz"

    print("=" * 60)
    print("Target transcription")
    print(true_transcription)

    # Print all decoding methods results
    for d_strategy in ["greedy", "beam", "beam_lm", "beam_lm_rescore"]:
        print("-" * 60)
        print(f"{d_strategy} decoding") 
        transcript = decoder.decode(audio_input, method=d_strategy)
        print(f"{transcript}")
        print(f"Character-level Levenshtein distance: {Levenshtein.distance(true_transcription, transcript.strip())}")


if __name__ == "__main__":
    
    test_samples = [
        ("./assignment_2/examples/sample1.wav", "IF YOU ARE GENEROUS HERE IS A FITTING OPPORTUNITY FOR THE EXERCISE OF YOUR MAGNANIMITY IF YOU ARE PROUD HERE AM I YOUR RIVAL READY TO ACKNOWLEDGE MYSELF YOUR DEBTOR FOR AN ACT OF THE MOST NOBLE FORBEARANCE"),
        ("./assignment_2/examples/sample2.wav", "AND IF ANY OF THE OTHER COPS HAD PRIVATE RACKETS OF THEIR OWN IZZY WAS UNDOUBTEDLY THE MAN TO FIND IT OUT AND USE THE INFORMATION WITH A BEAT SUCH AS THAT EVEN GOING HALVES AND WITH ALL THE GRAFT TO THE UPPER BRACKETS HE'D STILL BE ABLE TO MAKE HIS PILE IN A MATTER OF MONTHS"),
        ("./assignment_2/examples/sample3.wav", "GUESS A MAN GETS USED TO ANYTHING HELL MAYBE I CAN HIRE SOME BUMS TO SIT AROUND AND WHOOP IT UP WHEN THE SHIPS COME IN AND BILL THIS AS A REAL OLD MARTIAN DEN OF SIN"),
        ("./assignment_2/examples/sample4.wav", "IT WAS A TUNE THEY HAD ALL HEARD HUNDREDS OF TIMES SO THERE WAS NO DIFFICULTY IN TURNING OUT A PASSABLE IMITATION OF IT TO THE IMPROVISED STRAINS OF I DIDN'T WANT TO DO IT THE PRISONER STRODE FORTH TO FREEDOM"),
        ("./assignment_2/examples/sample5.wav", "MARGUERITE TIRED OUT WITH THIS LONG CONFESSION THREW HERSELF BACK ON THE SOFA AND TO STIFLE A SLIGHT COUGH PUT UP HER HANDKERCHIEF TO HER LIPS AND FROM THAT TO HER EYES"),
        ("./assignment_2/examples/sample6.wav", "AT THIS TIME ALL PARTICIPANTS ARE IN A LISTEN ONLY MODE"),
        ("./assignment_2/examples/sample7.wav", "THE INCREASE WAS MAINLY ATTRIBUTABLE TO THE NET INCREASE IN THE AVERAGE SIZE OF OUR FLEETS"),
        ("./assignment_2/examples/sample8.wav", "OPERATING SURPLUS IS A NON CAP FINANCIAL MEASURE WHICH IS DEFINED AS FULLY IN OUR PRESS RELEASE"),
    ]

    decoder = Wav2Vec2Decoder()

    _ = [test(decoder, audio_path, target) for audio_path, target in test_samples]
