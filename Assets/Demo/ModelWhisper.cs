using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using Unity.Sentis;
using UnityEngine;
using System.IO;
using System.Threading.Tasks;
using UnityEngine.Profiling;
using Stopwatch = System.Diagnostics.Stopwatch;
using System.Runtime.ExceptionServices;
using System.Threading;
using Unity.Profiling.LowLevel.Unsafe;
using Unity.Profiling;
using TMPro;
using UnityEngine.Networking;

public class ModelWhisper : MonoBehaviour
{
    public AudioClip storyClip;

    public string logMelSpectrogramModelName = "logmel-spectrogram.sentis";
    public string encoderModelName = "whisper-tiny-encoder.sentis";
    public string decoderModelName = "whisper-tiny-decoder.sentis";

    public string encoderModelAssetONNXName = "whisper-tiny-encoder";
    public string decoderModelAssetONNXName = "whisper-tiny-decoder";

    public string vocabName = "vocab.json";
    public WhisperLanguage speakerLanguage = WhisperLanguage.KOREAN;
    
    public enum WhisperLanguage
    {
        ENGLISH = 50259,
        KOREAN = 50264,
        JAPAN = 50266
    }

    public bool isReplay = false;
    
    private Worker _logMelSpectrogramWorker;
    private Worker _encoderWorker;
    private Worker _decoderWorker;
    
    const BackendType BACKEND = BackendType.GPUCompute;
    
    //Audio
    private AudioClip _audioClip;
    private const int MAX_RECORDING_TIME = 30;
    private const int AUDIO_SAMPLING_RATE = 16000;
    private const int maxSamples = MAX_RECORDING_TIME * AUDIO_SAMPLING_RATE;
    private int _numSamples;
    private float[] _data = new float[maxSamples];
    
    //Tokens
    private string[] _tokens;
    private int _currentToken = 0;
    private int[] _outputTokens = new int[maxTokens];
    
    // Used for special character decoding
    private int[] _whiteSpaceCharacters = new int[256];

    private Tensor _encodedAudio;

    private bool _transcribe = false;
    private string _outputString = "";

    const int maxTokens = 100;
    const int END_OF_TEXT = 50257;
    const int START_OF_TRANSCRIPT = 50258;
    const int TRANSCRIBE = 50359; //for speech-to-text in specified language
    const int TRANSLATE = 50358;  //for speech-to-text then translate to English
    const int NO_TIME_STAMPS = 50363; 
    const int START_TIME = 50364;

    ProfilerMarker markerencoderModelAssetONNX = new ProfilerMarker("Sentis.encoderModelAssetONNX");
    ProfilerMarker markerdecoderModelAssetONNX = new ProfilerMarker("Sentis.decoderModelAssetONNX");
    ProfilerMarker markerencoderModelSentis = new ProfilerMarker("Sentis.UDay.LoadEncoderModel");
    ProfilerMarker markerdecoderModelSentis = new ProfilerMarker("Sentis.UDay.LoadDecoderModel");

    public TMP_Text STTText;

    string GetPathFromStreamingAssets(string path)
    {
        #if UNITY_ANDROID && !UNITY_EDITOR
        var loadingRequest = UnityWebRequest.Get(Path.Combine(Application.streamingAssetsPath, path));
        loadingRequest.SendWebRequest();
        while (!loadingRequest.isDone)
        {
            if (loadingRequest.isNetworkError || loadingRequest.isHttpError)
            {
                break;
            }
        }
        if (loadingRequest.isNetworkError || loadingRequest.isHttpError)
        {
            return null;
        }
        else
        {
            File.WriteAllBytes(Path.Combine(Application.persistentDataPath , path), loadingRequest.downloadHandler.data);
            return Path.Combine(Application.persistentDataPath, path);
        }
        #else
        return Path.Combine(Application.streamingAssetsPath, path);
        #endif
    }

    void Start()
    {
        Model logMelSpectrogramModel = ModelLoader.Load(GetPathFromStreamingAssets(logMelSpectrogramModelName));

        /*
        markerencoderModelAssetONNX.Begin();
        ModelAsset encoderModelAssetONNX = Resources.Load<ModelAsset>(encoderModelAssetONNXName);
        Model encoderModelOnnx = ModelLoader.Load(encoderModelAssetONNX);
        markerencoderModelAssetONNX.End();

        markerdecoderModelAssetONNX.Begin();
        ModelAsset decoderModelAssetONNX = Resources.Load<ModelAsset>(decoderModelAssetONNXName);
        Model decoderModelOnnx = ModelLoader.Load(decoderModelAssetONNX);
        markerdecoderModelAssetONNX.End();
        */

        markerencoderModelSentis.Begin();
        Model encoderModel = ModelLoader.Load(GetPathFromStreamingAssets(encoderModelName));
        markerencoderModelSentis.End();

        markerdecoderModelSentis.Begin();
        Model decoderModel = ModelLoader.Load(GetPathFromStreamingAssets(decoderModelName));
        markerdecoderModelSentis.End();

        _logMelSpectrogramWorker = new Worker(logMelSpectrogramModel, BACKEND);
        _encoderWorker = new Worker(encoderModel, BACKEND);
        _decoderWorker = new Worker(decoderModel, BACKEND);


        GetTokens();
        SetupWhiteSpaceShifts();
        PrepareAudioClip(storyClip);
    }

    bool isWorking = false;
    bool itrStarted = false;
    IEnumerator itrSchedule;
    const int k_LayersPerFrame = 10;
    Tensor<int> tokensSoFar;

    async void Update()
    {
        if (!isWorking && _transcribe && _currentToken < _outputTokens.Length - 1)
        {
            isWorking = true;
            {
                if(itrStarted == false)
                {
                    itrStarted = true;

                    tokensSoFar = new Tensor<int>(new TensorShape(1, _outputTokens.Length), _outputTokens);
                    _decoderWorker.SetInput(0, tokensSoFar);
                    _decoderWorker.SetInput(1, _encodedAudio);

                    if (k_LayersPerFrame < 0)
                        _decoderWorker.Schedule();
                    else
                    {
                        itrSchedule = _decoderWorker.ScheduleIterable();
                    }
                    
                }

                if (k_LayersPerFrame > 0)
                {
                    int it = 0;
                    while (itrSchedule.MoveNext())
                    {
                        if (++it % k_LayersPerFrame == 0)
                        {
                            isWorking = false;
                            return;
                        }
                    }
                }

                var tokensPredictions = _decoderWorker.PeekOutput() as Tensor<int>;

                using var cpuPredictions = await tokensPredictions.ReadbackAndCloneAsync();
                //using var cpuPredictions = tokensPredictions.ReadbackAndClone();
                
                int ID = cpuPredictions[_currentToken];

                _outputTokens[++_currentToken] = ID;

                if (ID == END_OF_TEXT)
                {
                    _transcribe = false;
                    tokensSoFar.Dispose();
                    _outputString = GetUnicodeText(_outputString);
                    STTText.text = _outputString;
                    Debug.Log(_outputString);
                }
                else if (ID >= _tokens.Length)
                {
                    _outputString += $"(time={(ID - START_TIME) * 0.02f})";
                    Debug.Log(_outputString);
                }
                else
                {
                    _outputString += _tokens[ID];
                    Debug.Log(GetUnicodeText(_outputString));
                    STTText.text = GetUnicodeText(_outputString);
                }

                itrStarted = false;
            }

            isWorking = false;
        }
    }
    
    public void StartRecording()
    {
        if(_audioClip != null)
        {
            AudioClip.Destroy(_audioClip);
            _audioClip = null;
        }
        
        _audioClip = Microphone.Start(null, false, MAX_RECORDING_TIME, AUDIO_SAMPLING_RATE);
        Debug.Log("Recording started.");
    }

    public bool StopRecording()
    {
        Microphone.End(null);

        if (_audioClip != null)
        {
            PrepareAudioClip();
        }
        else
        {
            Debug.LogWarning("No audio clip recorded.");
            return false;
        }

        // Destroy the audio clip after loading to free up memory
        AudioClip.Destroy(_audioClip);
        _audioClip = null;

        Debug.Log("Recording stopped.");
        return true;
    }

    public void PrepareAudioClip(AudioClip storedAudioClip = null)
    {
        if (storedAudioClip != null)
        {
            _audioClip = storedAudioClip;
        }

        AudioSource audioSource = GetComponent<AudioSource>();
        audioSource.clip = _audioClip;
        if (isReplay)
            audioSource.Play();
        
        LoadAudioClip();
    }

    void LoadAudioClip()
    {
        LoadAudio();
        EncodeAudio();
        _transcribe = true;
        _outputString = "";
        
        Array.Fill(_outputTokens, 0);
        
        _outputTokens[0] = START_OF_TRANSCRIPT;
        _outputTokens[1] = (int)speakerLanguage; //ENGLISH;//GERMAN;//FRENCH;//
        _outputTokens[2] = TRANSCRIBE; //TRANSLATE;//TRANSCRIBE;
        _outputTokens[3] = NO_TIME_STAMPS;// START_TIME;//
        _currentToken = 3;
    }

    void LoadAudio()
    {
        if(_audioClip.frequency != AUDIO_SAMPLING_RATE)
        {
            Debug.Log($"The audio clip should have frequency 16kHz. It has frequency {_audioClip.frequency / 1000f}kHz");
            return;
        }

        _numSamples = _audioClip.samples;

        if (_numSamples > maxSamples)
        {
            Debug.Log($"The AudioClip is too long. It must be less than 30 seconds. This clip is {_numSamples/ _audioClip.frequency} seconds.");
            return;
        }

        //_data = new float[maxSamples];
        _audioClip.GetData(_data, 0);
    }

    void GetTokens()
    {
        var jsonText = File.ReadAllText(GetPathFromStreamingAssets(vocabName));
        var vocab = Newtonsoft.Json.JsonConvert.DeserializeObject<Dictionary<string, int>>(jsonText);
        _tokens = new string[vocab.Count];
        foreach(var item in vocab)
        {
            _tokens[item.Value] = item.Key;
        }
    }

    void EncodeAudio()
    {
        using var input = new Tensor<float>(new TensorShape(1, maxSamples), _data);

        _logMelSpectrogramWorker.Schedule(input);
        var spectroOutput = _logMelSpectrogramWorker.PeekOutput() as Tensor<float>;

        _encoderWorker.Schedule(spectroOutput);
        _encodedAudio = _encoderWorker.PeekOutput() as Tensor<float>;
    }
    
    // Translates encoded special characters to Unicode
    string GetUnicodeText(string text)
    {
        var bytes = Encoding.GetEncoding("ISO-8859-1").GetBytes(ShiftCharacterDown(text));
        return Encoding.UTF8.GetString(bytes);
    }

    string ShiftCharacterDown(string text)
    {
        string outText = "";

        foreach (char letter in text)
        {
            outText += ((int)letter <= 256) ? letter :
                (char)_whiteSpaceCharacters[(int)(letter - 256)];
        }
        return outText;
    }

    void SetupWhiteSpaceShifts()
    {
        for (int i = 0, n = 0; i < 256; i++)
        {
            if (IsWhiteSpace((char)i)) _whiteSpaceCharacters[n++] = i;
        }
    }

    bool IsWhiteSpace(char c)
    {
        return !((33 <= c && c <= 126) || (161 <= c && c <= 172) || (187 <= c && c <= 255));
    }

    private void OnDestroy()
    {
        _logMelSpectrogramWorker?.Dispose();
        _encoderWorker?.Dispose();
        _decoderWorker?.Dispose();
    }
}