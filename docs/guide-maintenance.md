# Maintaining the user guide

The editable source is [eCAT Guide.docx](../eCAT%20Guide.docx). The
[PDF](../eCAT%20Guide.pdf) is its rendered distribution copy. Keep both aligned
with the release branch, not an experimental checkout.

1. Check signatures in the public modules and option defaults using
   `e.describe_options(section, {"print": False, "return": True})`.
2. Update parameters, option tables, returns, examples and limitations together.
   Use heading styles and native Word equations; preserve internal links and
   the function finder. Keep the Default header shaded like other headers and
   give it enough width to avoid splitting a word.
3. Reconcile the [API reference](api_reference.md),
   [equations](analysis_equations.md), [citations](analysis_reading.md) and
   relevant tutorial notebooks. An optimizer success flag is not evidence of
   numerical convergence or mechanistic validity.
4. Export the document to PDF using Word or LibreOffice. For a headless build,
   use a temporary output folder and a separate LibreOffice user profile:

   ```bash
   soffice -env:UserInstallation=file:///tmp/ecat-guide-profile \
     --headless --convert-to pdf --outdir /tmp/ecat-guide-render "eCAT Guide.docx"
   ```

5. Verify every PDF page for clipped text, table alignment, equations and
   navigation. Inspect dense option tables and modified pages at full size.
   Check internal Word bookmarks, PDF link destinations, Markdown links,
   Python examples and current signatures before replacing the public PDF.
6. Keep render images, caches and temporary environments outside synced source
   folders. Preserve source documents and final outputs. Do not include private
   measurements, research notebooks, credentials or custom backend builds in a
   release merely because they share the checkout.
