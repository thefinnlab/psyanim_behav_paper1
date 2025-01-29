import Phaser from 'phaser';

import PsyanimScene from './PsyanimScene.js';
import PsyanimDataDrivenScene from './PsyanimDataDrivenScene.js';

import {
    PsyanimDebug
} from 'psyanim-utils';

/**
 *  Application-wide configuration class for PsyanimApp
 */
export default class PsyanimConfig {

    constructor() {

        this._phaserConfig = {
            type: Phaser.AUTO,
            parent: 'phaser-app',
            width: 800,
            height: 600,
            backgroundColor: 0xffffff,
            fps: {
                target: 60,
                forceSetTimeOut: true
            },
            scene: [],
            physics: {
                default: 'matter',
                matter: {
                    gravity: {
                        y: 0
                    }
                }
            }
        };

        this._sceneKeys = [];

        this._dataDrivenScenes = [];
    }

    /**
     * Registers a scene so it is available to load at runtime (after `PsyanimApp.Instance.run()` is called).
     * 
     * @param {*} scene - either a `PsyanimScene type` or a `Psyanim Scene Definition`
     */
    registerScene(scene) {

        if (scene.prototype instanceof PsyanimScene)
        {
            if (!scene.hasOwnProperty('KEY'))
            {
                PsyanimDebug.error("Scene has no 'key' field!");
            }

            if (!this._phaserConfig.scene.includes(scene))
            {
                this._phaserConfig.scene.push(scene);
                this._sceneKeys.push(scene.KEY);
            }
            else
            {
                PsyanimDebug.error("ERROR: scene is already registered in app config!");
            }    
        }
        else
        {
            if (!Object.hasOwn(scene, 'key'))
            {
                PsyanimDebug.error('Scene has no key!');
            }

            this._dataDrivenScenes.push(scene);
            this._sceneKeys.push(scene.key);

            // if user registers at least 1 data-driven scene, we need to register PsyanimDataDrivenScene
            if (!this._phaserConfig.scene.includes(PsyanimDataDrivenScene))
            {
                this._phaserConfig.scene.push(PsyanimDataDrivenScene);
            }
        }
    }

    /**
     *  The keys currently registered with this config object.
     *  @type {string[]}
     */
    get sceneKeys() {

        return this._sceneKeys.slice();
    }

    /**
     *  A list of the `Psyanim scene definition`s currently registered with this config object.
     *  @type {string[]}
     */
    get sceneDefinitions() {

        return this._dataDrivenScenes.slice();
    }

    /**
     * Returns the scene definition associated with `key`.
     * 
     * @param {string} key - key that uniquely identifies a particular scene definition
     * @returns {Psyanim Scene Definition}
     */
    getSceneDefinition(key) {

        return this._dataDrivenScenes.find(s => s.key === key);
    }

    /**
     * Enables / disables application-wide debug features, including Phaser's matter-js debug visualizations.
     * 
     * @param {boolean} enabled
     */
    setDebugEnabled(enabled) {

        if (enabled)
        {
            this._phaserConfig.physics.matter.debug = {

                showAxes: false,
                showAngleIndicator: true,
                angleColor: 0xe81153,

                showBroadphase: false,
                broadphaseColor: 0xffb400,

                showBounds: false,
                boundsColor: 0xffffff,

                showVelocity: true,
                velocityColor: 0x00aeef,

                showCollisions: false,
                collisionColor: 0xf5950c,
    
                showSeparations: false,
                separationColor: 0xffa500,

                showBody: true,
                showStaticBody: true,
                showInternalEdges: true,

                renderFill: false,
                renderLine: true,
    
                fillColor: 0x106909,
                fillOpacity: 1,
                lineColor: 0x28de19,
                lineOpacity: 1,
                lineThickness: 1,
    
                staticFillColor: 0x0d177b,
                staticLineColor: 0x1327e4,

                showSleeping: true,
                staticBodySleepOpacity: 1,
                sleepFillColor: 0x464646,
                sleepLineColor: 0x999a99,
    
                showSensors: true,
                sensorFillColor: 0x0d177b,
                sensorLineColor: 0x1327e4,
    
                showPositions: true,
                positionSize: 4,
                positionColor: 0xe042da,
    
                showJoint: true,
                jointColor: 0xe0e042,
                jointLineOpacity: 1,
                jointLineThickness: 2,
    
                pinSize: 4,
                pinColor: 0x42e0e0,
    
                springColor: 0xe042e0,
    
                anchorColor: 0xefefef,
                anchorSize: 4,
    
                showConvexHulls: true,
                hullColor: 0xd703d0
            };
        }
        else
        {
            this._phaserConfig.physics.matter.debug = false;
        }
    }

    /**
     *  The underlying Phaser config object managed by this PsyanimConfig object.
     * 
     *  For full API docs of Phaser config object, see https://newdocs.phaser.io/docs/3.70.0/Phaser.Core.Config
     * 
     *  @type {Object}
     */
    get phaserConfig() {

        return this._phaserConfig;
    }
}