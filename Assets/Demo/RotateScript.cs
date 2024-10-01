using UnityEngine;

public class RotateScript : MonoBehaviour
{
    void Update()
    {
        this.transform.eulerAngles += 1.0f * Vector3.up;
    }
}
