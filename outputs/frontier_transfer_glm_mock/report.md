==============================================================================
FRONTIER-LLM TRANSFER TEST - SUMMARY
==============================================================================

Images tested  : 10
Models         : glm
Wall-clock (s) : 3.2

Image                                  glm
------------------------------------------
adv_dog                        no transfer
adv_dog01                      no transfer
adv_dog02                      no transfer
adv_dog03                      no transfer
adv_dog04                      no transfer
adv_dog05                      no transfer
adv_dog06                      no transfer
adv_dog07                      no transfer
adv_dog08                      no transfer
adv_dog09                      no transfer

Per-model transfer rate (clean->no-dog):
  glm        transfer=0/10 (  0.0%)  clean_dog=10  adv_dog=10

Interpretation:
  transfer success  -> frontier model now fails to say 'dog' on the
                       adversarial image but would have said 'dog' on
                       the clean original (humans still see a dog).
  no transfer       -> frontier model still says 'dog'; perturbation
                       did not survive the transfer.
