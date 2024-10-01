using UnityEngine;
using UnityEngine.XR;
using TMPro;

public class FrameTest : MonoBehaviour
{
    public TMP_Text fpsText;
    float deltaTime = 0.0f;
 
    void Start()
    {
        //Application.targetFrameRate = 30;
    }
    
    async void Update()
    {
        deltaTime += (Time.unscaledDeltaTime - deltaTime) * 0.1f;
        float msec = deltaTime * 1000.0f;
        float fps = 1.0f / deltaTime;
        fpsText.text = string.Format("{0:0.0} ms ({1:0.} fps)", msec, fps);
    }
}
