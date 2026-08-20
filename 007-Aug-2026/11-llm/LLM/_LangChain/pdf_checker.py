# write a code to all the text and images of the 'company_policy.pdf' file into a database using langchain and huggingface embeddings. The code should read the pdf, split it into chunks, create embeddings for each chunk, and store them in a FAISS database.

import fitz # PyMuPDF need to be installed for image extraction

doc = fitz.open("company_policy.pdf")
for page in doc:
    text = page.get_text()
    print(text)  # Print the text of each page

    # Extract images from the page
    image_list = page.get_images(full=True)
    for img_index, img in enumerate(image_list):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]
        
        # Save the image to a file
        image_filename = f"page_{page.number + 1}_image_{img_index + 1}.{image_ext}"
        with open(image_filename, "wb") as img_file:
            img_file.write(image_bytes)
        
        print(f"Extracted image saved as: {image_filename}")