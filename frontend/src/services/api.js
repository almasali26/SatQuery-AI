const API_URL = 'http://127.0.0.1:8000'

export async function analyzeImage(image, question) {
  if (!image?.file) {
    throw new Error('No image file was selected.')
  }

  if (!question?.trim()) {
    throw new Error('Please enter a question about the image.')
  }

  const formData = new FormData()

  formData.append('image', image.file)
  formData.append('question', question.trim())

  const response = await fetch(`${API_URL}/analyze`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    let message = 'The backend could not analyze this image.'

    try {
      const errorData = await response.json()

      if (errorData?.detail) {
        message = errorData.detail
      }
    } catch {
      // Keep the default error message
    }

    throw new Error(message)
  }

  return await response.json()
}

export async function compareImages(image1, image2) {
  if (!image1?.file || !image2?.file) {
    throw new Error('Please select two images for comparison.')
  }

  const formData = new FormData()

  formData.append('image1', image1.file)
  formData.append('image2', image2.file)

  const response = await fetch(`${API_URL}/compare`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    let message = 'The backend could not compare these images.'

    try {
      const errorData = await response.json()

      if (errorData?.detail) {
        message = errorData.detail
      }
    } catch {
      // Keep the default error message
    }

    throw new Error(message)
  }

  return await response.json()
}