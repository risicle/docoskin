# Docoskin

"Onion-skin" visual differences between a reference document image and a scanned copy.

Given an image of a reference document, docoskin will attempt to find key points of the document in a scanned copy,
align the two, auto-correct the image contrasts and display a combined image, featuring sections in red which were
"removed from" the reference document and sections in green which were added to the document.

The intended use for this is comparing a signed, returned scan of a contract with the original version as provided to
the signer to check for unagreed amendments.

Implemented as a python library with a command-line interface. Significantly more power and flexibility is
achievable through accessing the python components.

Depends on opencv and six (and, on python2.7, the `futures` backport).

Complete packaging is a work-in-progress. Preliminary license is GPLv3.
