"""A regular package on purpose: linkpreview (pulled in by neonize, for WhatsApp)
installs its own top-level `tests` package, which hid this folder as a namespace
package -- `python3 -m tests.test_...` then failed with "No module named"."""
