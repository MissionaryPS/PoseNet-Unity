using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WebCamTest : MonoBehaviour
{
    

    // Start is called before the first frame update
    void Start()
    {
        Debug.Log(WebCamTexture.devices.Length);    
        if(WebCamTexture.devices.Length > 0)
        {
            foreach(var device in WebCamTexture.devices)
            {
                Debug.Log(device.name);
            }
        }
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
