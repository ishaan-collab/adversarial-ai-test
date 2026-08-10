==============================================================================
FRONTIER-LLM TRANSFER TEST - SUMMARY
==============================================================================

Images tested  : 0
Models         : glm5
Wall-clock (s) : 0.0

Image                                 glm5
------------------------------------------

Per-model transfer rate (clean->no-dog):

Interpretation:
  transfer success  -> frontier model now fails to say 'dog' on the
                       adversarial image but would have said 'dog' on
                       the clean original (humans still see a dog).
  no transfer       -> frontier model still says 'dog'; perturbation
                       did not survive the transfer.
