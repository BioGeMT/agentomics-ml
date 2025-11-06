from test.utils_test import BaseAgentTest, check_foundation_model_gpu_usage

class TestFoundationModels(BaseAgentTest):
    """Test suite for foundation model usage in agents"""

    def test_agent_access_docs(self):
        """Test that agent can access foundation models documentation."""

        doc_files = [
            "ESM-2.md",
            "HyenaDNA.md",
            "DNABERT-2.md",
            "NucleotideTransformer.md"
        ]

        for doc_file in doc_files:
            with self.subTest(doc_file=doc_file): #checks which file fails (if any)
                result = self.bash_tool.function(f"head -5 /foundation_models/{doc_file} 2>&1")
                self.assertNotIn("Permission denied" or "No such file", result, "Agent should read foundation model documentation")
                self.assertNotIn("No such file", result, f"{doc_file} should exist in foundation models directory")
                self.assertNotIn("Command failed", result, "head command should succeed")

    def test_huggingface_env_var(self):
        """Test that agent has HF_HOME set correctly."""

        result = self.bash_tool.function("echo $HF_HOME 2>&1")
        self.assertIn("/foundation_models", result, "HF_HOME should point to /foundation_models")
        
    def test_agent_list_models(self):
        """Test that agent can list foundation models directory."""

        result = self.bash_tool.function("ls /foundation_models/hub 2>&1")
        self.assertNotIn("Permission denied" or "No such file", result, "Agent should list foundation models directory")
        self.assertIn("esm2", result.lower(), "ESM-2 directory should exist in foundation models")
        self.assertIn("hyenadna", result.lower(), "HyenaDNA directory should exist in foundation models")
        self.assertIn("nucleotide-transformer", result.lower(), "NucleotideTransformer directory should exist in foundation models")

    def test_agent_esm2_usage(self):
        """Test that agent can use ESM-2 model for protein embeddings."""
                                                                                                                       
        esm2_code = """                                                                                                  
import torch                                                                                                         
from transformers import AutoTokenizer, AutoModel                                                                    

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU available, using CPU.")
                                                                                                                    
# Load ESM-2 model                                                                                                   
model_name = "facebook/esm2_t6_8M_UR50D"                                                                             
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)                 
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, local_files_only=True).to(device)                    
                                                                                                                    
# Test inference on protein sequence
protein = "MKTII"
inputs = tokenizer(protein, return_tensors='pt').to(device)
outputs = model(**inputs)
hidden_states = outputs.last_hidden_state

print(f"Success! Output shape: {hidden_states.shape}")
  """
        test_file_path = self.config.runs_dir / self.config.agent_id / "esm2_test.py"
        self.write_python_tool.function(file_path=test_file_path, code=esm2_code)

        run_result = self.run_python_tool.function(python_file_path=test_file_path)
        self.assertNotIn("error", run_result.lower(), "ESM-2 script should run without errors")
        self.assertIn("Success!", run_result, "ESM-2 should produce output")

        check_foundation_model_gpu_usage(run_result, model="ESM-2")

    def test_agent_hyena_dna_usage(self):
        """Test that agent can use HyenaDNA model for DNA sequence modeling."""
                                                                                                                       
        hyena_dna_code = """
import torch
from transformers import AutoTokenizer, AutoModel

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU available, using CPU.")
    
# Load HyenaDNA model (using tiny model for faster testing)                                                                                              
model_name = "LongSafari/hyenadna-tiny-1k-seqlen-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, local_files_only=True).to(device)

# Test inference on DNA sequence
dna_sequence = "ACTGACTG" * 100  # Short sequence for testing
inputs = tokenizer(dna_sequence, return_tensors='pt').to(device)
outputs = model(**inputs)
hidden_states = outputs.last_hidden_state

print(f"Success! Output shape: {hidden_states.shape}")
"""

        test_file_path = self.config.runs_dir / self.config.agent_id / "hyena_dna_test.py"
        self.write_python_tool.function(file_path=test_file_path, code=hyena_dna_code)

        run_result = self.run_python_tool.function(python_file_path=test_file_path)
        self.assertNotIn("error", run_result.lower(), "HyenaDNA script should run without errors")
        self.assertIn("Success!", run_result, "HyenaDNA should produce output")

        check_foundation_model_gpu_usage(run_result, model="HyenaDNA")

    def test_agent_nucleotide_transformer_usage(self):
        """Test that agent can use NucleotideTransformer model for genomic sequences."""

        nt_code = """
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU available, using CPU.")

# Load NucleotideTransformer model (using 50M model for faster testing)
model_name = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True, local_files_only=True).to(device)

# Test inference on DNA sequences
sequences = ["ATTCCGATTCCGATTCCG", "ATTTCTCTCTCTCTCTGAGATCGATCGATCGAT"]
max_length = min(tokenizer.model_max_length, 512)  # Use smaller length for testing

tokens_ids = tokenizer.batch_encode_plus(
    sequences, return_tensors="pt", padding="max_length", max_length=max_length
)["input_ids"].to(device)

attention_mask = tokens_ids != tokenizer.pad_token_id
torch_outs = model(
    tokens_ids,
    attention_mask=attention_mask,
    encoder_attention_mask=attention_mask,
    output_hidden_states=True
)

embeddings = torch_outs['hidden_states'][-1]

print(f"Success! Embeddings shape: {embeddings.shape}")
"""
        test_file_path = self.config.runs_dir / self.config.agent_id / "nucleotide_transformer_test.py"
        self.write_python_tool.function(file_path=test_file_path, code=nt_code)

        run_result = self.run_python_tool.function(python_file_path=test_file_path)
        self.assertNotIn("error", run_result.lower(), "NucleotideTransformer script should run without errors")
        self.assertIn("Success!", run_result, "NucleotideTransformer should produce output")

        check_foundation_model_gpu_usage(run_result, model="NucleotideTransformer")

    def test_agent_rinalmo_usage(self):
        """Test that agent can use RiNALMo model for RNA sequence modeling."""

        rinalmo_code = """
import torch
from multimolecule import RnaTokenizer, RiNALMoModel

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU available, using CPU.")

# Load RiNALMo model (using micro model for faster testing)
model_name = "multimolecule/rinalmo-micro"
tokenizer = RnaTokenizer.from_pretrained(model_name, local_files_only=True)
model = RiNALMoModel.from_pretrained(model_name, local_files_only=True).to(device)

# Test inference on RNA sequence
rna = "UAGCUUAUCAGACUGAUGUUG"
inputs = tokenizer(rna, return_tensors="pt")
# Move inputs to device
inputs = {k: v.to(device) for k, v in inputs.items()}
outputs = model(**inputs)
embeddings = outputs.last_hidden_state

print(f"Success! Embeddings shape: {embeddings.shape}")
"""
        test_file_path = self.config.runs_dir / self.config.agent_id / "rinalmo_test.py"
        self.write_python_tool.function(file_path=test_file_path, code=rinalmo_code)

        run_result = self.run_python_tool.function(python_file_path=test_file_path)
        self.assertNotIn("error", run_result.lower(), "RiNALMo script should run without errors")
        self.assertIn("Success!", run_result, "RiNALMo should produce output")

        check_foundation_model_gpu_usage(run_result, model="RiNALMo")