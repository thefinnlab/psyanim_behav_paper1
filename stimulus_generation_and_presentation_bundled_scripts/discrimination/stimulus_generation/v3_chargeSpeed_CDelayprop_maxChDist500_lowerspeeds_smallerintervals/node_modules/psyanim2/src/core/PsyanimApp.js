import Phaser from 'phaser';

import PsyanimConfig from './PsyanimConfig.js';
import PsyanimDataDrivenScene from './PsyanimDataDrivenScene.js';

import EmptyScene from './scene_templates/EmptyScene.js';

import {
    PsyanimDebug
} from 'psyanim-utils';

import { v4 as uudiv4 } from 'uuid';

/**
 *  PsyanimApp is a Singleton class that encapsulates the entire real-time application for Psyanim2.
 * 
 *  PsyanimApp contains the Phaser.Game instance and manages the parent DOM element for the Phaser HTML5 canvas.
 */
export default class PsyanimApp {

    /**
     *  Application-wide Singleton Instance of PsyanimApp
     *  @type {PsyanimApp}
     */
    static get Instance() {

        if (PsyanimApp._instance == null)
        {
            PsyanimApp._instance = new PsyanimApp();
        }

        return PsyanimApp._instance;
    }

    /**
     *  Application-wide event emitter for PsyanimApp.
     *  @type {Phaser.Events.EventEmitter}
     */
    events;

    constructor() {

        if (PsyanimApp._instance != null)
        {
            PsyanimDebug.error('Attempting to instantiate another instance of PsyanimApp, but PsyanimApp is a singleton!');
        }

        this._config = new PsyanimConfig();

        window.psyanimApp = this;

        this._sessionID = uudiv4();

        this.events = new Phaser.Events.EventEmitter();
    }

    /**
     * Toggles canvas visibility by appending/removing HTML element to/from it's parent element in the document.body.
     * 
     * @param {boolean} visible 
     */
    setCanvasVisible(visible) {

        if (visible)
        {
            let canvas = this._game.canvas;

            if (canvas.parentElement == null)
            {
                this._domElement.appendChild(canvas);
            }
        }
        else
        {
            this._domElement.innerHTML = '';
        }
    }

    /**
     *  The session ID for this app instance.  Guaranteed to be unique across all PsyanimApp instances.
     *  @type {string} 
     */
    get sessionID() {
        return this._sessionID;
    }

    /**
     *  The current PsyanimScene that is running.
     *  @type {PsyanimScene}
     */
    get currentScene() {

        return this._game.registry.get('psyanim_currentScene');
    }

    /**
     *  The scene key of the current scene that is running.
     *  @type {string}
     */
    get currentSceneKey() {

        return this._currentSceneKey;
    }

    /**
     *  All configured scene keys.
     *  @type {string[]}
     */
    get sceneKeys() {

        return this._config.sceneKeys;
    }

    /**
     *  The parent element of the Phaser canvas.
     *  @type {object}
     */
    get domElement() {

        return this._domElement;
    }

    /**
     *  The PsyanimConfig used for this PsyanimApp.Instance.
     *  @type {PsyanimConfig}
     */
    get config() {

        return this._config;
    }

    /**
     *  The Phaser.Game instance for this PsyanimApp.Instance.  This is `undefined` until you PsyanimApp.Instance.run() is called.
     *  @type {Phaser.Game}
     */
    get game() {

        return this._game;
    }

    /**
     * Returns the PsyanimScene associated with `sceneKey`
     * 
     * @param {string} sceneKey 
     * @returns {PsyanimScene}
     */
    getSceneByKey(sceneKey) {

        if (!this.sceneKeys.includes(sceneKey))
        {
            PsyanimDebug.error('No scene with key: ' + sceneKey);
        }

        let psyanimScenes = this.config.phaserConfig.scene;
        let psyanimSceneKeys = psyanimScenes.map(s => s.KEY);

        if (psyanimSceneKeys.includes(sceneKey))
        {
            // return the actual psyanim scene reference
            return PsyanimApp.Instance.currentScene.scene.get(sceneKey);
        }
        else
        {
            // return the data-driven scene b.c. it will be the one used for this key
            return PsyanimApp.Instance.currentScene.scene.get(PsyanimDataDrivenScene.KEY);
        }
    }

    /**
     * Load a particular scene by key.
     * 
     * NOTE: Scene must have been registered with PsyanimApp.Instance.config
     * 
     * @param {string} sceneKey - key identifying a particular scene to be loaded.
     */
    loadScene(sceneKey) {

        PsyanimDebug.log("load scene called with key: " + sceneKey);

        if (!this.sceneKeys.includes(sceneKey))
        {
            PsyanimDebug.error("Failed to load scene '" + sceneKey + "' - no scene registered with that key!");
        }

        let psyanimScenes = this.config.phaserConfig.scene;
        let psyanimSceneKeys = psyanimScenes.map(s => s.KEY);

        if (psyanimSceneKeys.includes(sceneKey))
        {
            this.currentScene.scene.start(sceneKey);
        }
        else
        {
            let sceneDefinition = this._config.getSceneDefinition(sceneKey);

            this._game.registry.set('psyanim_currentSceneDefinition', sceneDefinition);

            this.currentScene.scene.start(PsyanimDataDrivenScene.KEY);
        }

        this._currentSceneKey = sceneKey;
    }

    /**
     *  Stops the current scene from running.
     */
    stopCurrentScene() {

        let currentScene = this.currentScene;

        currentScene.scene.stop(currentScene.KEY);
    }

    _loadExperimentVariations(experimentDefinition) {

        let experimentVariations = [];

        for (let i = 0; i < experimentDefinition.runs.length; ++i)
        {
            let run = experimentDefinition.runs[i];

            if (!run.sceneType.KEY)
            {
                PsyanimDebug.error("ERROR: to register a PsyanimScene type, it must contain a static string field called 'KEY'!");
            }

            if (!this._config.phaserConfig.scene.includes(run.sceneType))
            {
                this._config.phaserConfig.scene.push(run.sceneType);
            }

            for (let j = 0; j < run.variations; ++j)
            {
                experimentVariations.push({
                    sceneKey: run.sceneType.KEY,
                    parameterSet: run.parameterSet,
                    runNumber: i,
                    variationNumber: j,
                    agentNamesToRecord: run.agentNamesToRecord
                });
            }
        }

        return experimentVariations;
    }

    /**
     *  Current player ID for this session.
     *  @type {string}
     */
    get currentPlayerID() {

        return this._game.registry.get('psyanim_currentPlayerID');
    }

    /**
     * Sets the current player ID for this session.
     * @type {string}
     */
    set currentPlayerID(value) {

        this._game.registry.set('psyanim_currentPlayerID', value);
    }

    /**
     *  Starts PsyanimApp, creating a new Phaser.Game() and runs the first scene in `this.sceneKeys`.
     */
    run() {

        if (this.sceneKeys.length == 0)
        {
            console.warn('No scenes configured - adding empty scene to config!');

            this.config.registerScene(EmptyScene);
        }

        // create phaser game
        this._game = new Phaser.Game(this._config.phaserConfig);

        // setup canvas dom parent
        this._domElement = document.getElementById('phaser-app');

        if (this._domElement == null)
        {
            this._domElement = document.createElement('div');
            this._domElement.setAttribute('id', 'phaser-app');
            
            document.body.appendChild(this._domElement);
        }

        // setup canvas CSS class
        if (!this._domElement.classList.contains('phaser-canvas'))
        {
            this._domElement.classList.add('phaser-canvas');
        }

        // setup current scene info
        this._currentSceneKey = this.sceneKeys[0];

        let sceneDefinitionKeys = this.config.sceneDefinitions.map(s => s.key);

        if (sceneDefinitionKeys.includes(this._currentSceneKey))
        {
            let sceneDefinition = this.config.getSceneDefinition(this._currentSceneKey);

            this._game.registry.set('psyanim_currentSceneDefinition', sceneDefinition);
        }
    }
}

PsyanimApp._instance = null;