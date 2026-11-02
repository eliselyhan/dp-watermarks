"""Fixed dimensions of the bundled distinct-prompt experiment."""

# Counts per prompt. Candidate blocks yield one watermarked response each.
# These dimensions describe pool construction in distinct_prompts.replay;
# that module samples mixtures, and distinct_prompts.main estimates their share.
COMPLETIONS_PER_PROMPT = 9_992
CANDIDATES_PER_BLOCK = 20
WATERMARKED_POOL_SIZE = 450
CANDIDATE_POOL_SIZE = WATERMARKED_POOL_SIZE * CANDIDATES_PER_BLOCK
BASE_POOL_SIZE = COMPLETIONS_PER_PROMPT - CANDIDATE_POOL_SIZE

NUM_PROMPTS = 100
MAX_TOKENS = 20
MIXTURE_SIZE = 500
REPEATS_PER_KEY = 5_000
LAMBDA_TENTHS = (1, 5, 9)
PROMPT_COUNTS = (1, 5, 10, 20, 50, 100)
