import PsyanimApp from '../core/PsyanimApp.js';
import PsyanimJsPsychPlugin from './PsyanimJsPsychPlugin.js';
import PsyanimJsPsychTrialParameter from './PsyanimJsPsychTrialParameter.js';

import {
    PsyanimUtils,
    PsyanimDebug
} from 'psyanim-utils';

/**
 * PsyanimJsPsychTrial class provides an API for building up a trial object as required
 * by the PsyanimJsPsychPlugin, for ease-of-use and to serve as a form of documentation itself.
 * 
 * It is highly recommended to use this class to generate the jsPsych trial object for your scene, 
 * rather than building it up manually.
 * 
 * It is also recommmended that you construct a new PsyanimJsPsychTrial object for each trial
 * in your jsPsych experiment.
 */
export default class PsyanimJsPsychTrial {

    /**
     * @param {Object} sceneDefinitionTemplate - the Psyanim scene definition which will be used as a template for this trial
     * @param {string} [sceneKey] - if provided, the scene key used for this trial will be overriden with this field's value
     */
    constructor(sceneDefinitionTemplate, sceneKey = null) {

        this._sceneDefinition = PsyanimUtils.cloneSceneDefinition(sceneDefinitionTemplate);

        // TODO: verify scene key is unique

        if (sceneKey)
        {
            this._sceneDefinition.key = sceneKey;
        }

        PsyanimApp.Instance.config.registerScene(this._sceneDefinition);

        this._validEntityParameters = ['initialPosition', 'shapeParams', 'matterOptions', 'instances'];

        this._validShapeParameters = [
            'shapeType', 'radius', 'base', 'altitude', 'width', 'height', 'depth', 'color'
        ];

        this._validMatterOptions = [
            'isSensor', 'isStatic'
        ];

        // setup other plugin parameters
        this._endTrialKeys = ['enter'];
        this._duration = -1.0;
        this._endTrialOnContact = false;
        this._endTrialOnPlaybackComplete = false;
        this._agentNamesToRecord = [];
        this._saveTrialMetadata = true;
        this._recordAnimationClips = true;
        this._recordStateLogs = true;
        this._subtext = '';
        this._jsPsychData = {};
        this._extensions = [];

        // trial parameters that the user can decide to save
        this._trialParametersToSave = [];

        // animation parameters that the user can decide to save
        this._animationParameters = [];
    }

    /**
     * Returns the `jsPsych trial object` corresponding to this trial's configuration for your `jsPsych timeline`.
     * 
     * @return {Object} - jsPsych trial object, to be added to jsPsych timeline
     */
    get jsPsychTrialDefinition() {

        return {
            type: PsyanimJsPsychPlugin,
            sceneKey: this.sceneKey,
            duration: this.duration,
            endTrialOnContact: this.endTrialOnContact,
            endTrialOnPlaybackComplete: this.endTrialOnPlaybackComplete,
            agentNamesToRecord: this.agentNamesToRecord,
            saveTrialMetadata: this._saveTrialMetadata,
            recordAnimationClips: this.recordAnimationClips,
            recordStateLogs: this.recordStateLogs,
            subtext: this._subtext,
            trialParameters: this._trialParametersToSave,
            animationParameters: this._animationParameters,
            data: this._jsPsychData,
            extensions: this._extensions
        };
    }

    /**
     * The scene key associated with this trial
     */
    get sceneKey() {

        return this._sceneDefinition.key;
    }

    /**
     * The `duration` parameter for a trial determines the maximum amount of time the trial should last.
     * 
     * @return {number} - trial duration parameter
     */
    get duration() {

        return this._duration;
    }

    /**
     * The `duration` parameter for a trial determines the maximum amount of time the trial should last.
     * 
     * @param {number} - trial duration parameter
     */
    set duration(value) {

        this._duration = value;
    }

    /**
     * 
     */
    get endTrialOnContact() {

        return this._endTrialOnContact;
    }

    set endTrialOnContact(value) {

        this._endTrialOnContact = value;
    }

    get endTrialOnPlaybackComplete() {

        return this._endTrialOnPlaybackComplete;
    }

    set endTrialOnPlaybackComplete(value) {

        this._endTrialOnPlaybackComplete = value;
    }

    get agentNamesToRecord() {

        return this._agentNamesToRecord;
    }

    get jsPsychData() {

        return this._jsPsychData;
    }

    set jsPsychData(value) {

        this._jsPsychData = value;
    }

    addExtension(type, params = {}) {

        // TODO: can we do some error-checking?

        this._extensions.push({ 
            type: type,
            params: params
        });
    }

    addAgentNamesToRecord(namesArray) {

        if (!Array.isArray(namesArray))
        {
            PsyanimDebug.error("'namesArray' must be an array of strings!");
            return;
        }

        this._agentNamesToRecord = this._agentNamesToRecord.concat(namesArray);
    }

    get saveTrialMetadata() {

        return this._saveTrialMetadata;
    }

    set saveTrialMetadata(value) {

        this._saveTrialMetadata = value;
    }

    get recordAnimationClips() {

        return this._recordAnimationClips;
    }

    set recordAnimationClips(value) {

        this._recordAnimationClips = value;
    }

    get recordStateLogs() {

        return this._recordStateLogs;
    }

    set recordStateLogs(value) {

        this._recordStateLogs = value;
    }

    get subtext() {

        return this._subtext;
    }

    set subtext(value) {

        this._subtext = value;
    }

    _getEntityDefinition(entityName) {

        let entityDefinition =  this._sceneDefinition.entities.find(e => e.name === entityName);

        if (!entityDefinition)
        {
            PsyanimDebug.error('No entity definition for entity named: ' + entityName);
        }

        return entityDefinition;
    }

    setEntityParameter(entityName, parameterName, value) {

        let entityDefinition = this._getEntityDefinition(entityName);

        if (!this._validEntityParameters.includes(parameterName))
        {
            PsyanimDebug.error("Parameter name '" + parameterName + "' is not a valid entity parameter. " + 
                "Valid parameters are: " + JSON.stringify(this._validEntityParameters));

            return;
        }

        entityDefinition[parameterName] = value;
    }

    setEntityShapeParameter(entityName, parameterName, value) {

        let entityDefinition = this._getEntityDefinition(entityName);

        if (!Object.hasOwn(entityDefinition, 'shapeParams'))
        {
            entityDefinition['shapeParams'] = {};
        }

        let shapeParams = entityDefinition['shapeParams'];

        if (!this._validShapeParameters.includes(parameterName))
        {
            PsyanimDebug.error("Shape parameter name '" + parameterName + "' is not a valid shape parameter. " + 
                "Valid parameters are: " + JSON.stringify(this._validShapeParameters));

            return;
        }

        shapeParams[parameterName] = value;
    }

    setEntityMatterOptions(entityName, parameterName, value) {

        let entityDefinition = this._getEntityDefinition(entityName);

        if (!Object.hasOwn(entityDefinition, 'matterOptions'))
        {
            entityDefinition['matterOptions'] = {};
        }

        let matterOptions = entityDefinition['matterOptions'];

        if (!this._validMatterOptions.includes(parameterName))
        {
            PsyanimDebug.error("Matter options parameter name '" + parameterName + "' is not a valid parameter. " + 
                "Valid parameters are: " + JSON.stringify(this._validMatterOptions));

            return;
        }

        matterOptions[parameterName] = value;
    }

    setPrefabParameter(entityName, parameterName, value, saveParameter = true) {

        let prefabParams = this.getPrefabParameters(entityName);

        prefabParams[parameterName] = value;

        if (saveParameter)
        {
            this.savePrefabParameter(entityName, parameterName);
        }
    }

    setComponentParameter(entityName, componentType, parameterName, value, saveParameter = true) {

        let componentParameters = this.getComponentParameters(entityName, componentType);

        componentParameters[parameterName] = value;

        if (saveParameter)
        {
            this.saveComponentParameter(entityName, componentType, parameterName);
        }
    }

    getSceneParameter(parameterName) {

        // TODO: validate parameter name

        return this._sceneDefinition[parameterName];
    }

    getEntityParameter(entityName, parameterName) {

        let entityDefinition = this._getEntityDefinition(entityName);

        if (!this._validEntityParameters.includes(parameterName))
        {
            PsyanimDebug.error("Parameter name '" + parameterName + "' is not a valid entity parameter. " + 
                "Valid parameters are: " + JSON.stringify(this._validEntityParameters));

            return;
        }

        return entityDefinition[parameterName];
    }

    getPrefabParameters(entityName) {

        let entityDefinition = this._getEntityDefinition(entityName);

        if (!Object.hasOwn(entityDefinition, 'prefab'))
        {
            PsyanimDebug.error('No prefab field found for entity named: ' + entityName);
            return;
        }

        let prefabDefinition = entityDefinition['prefab'];

        if (!Object.hasOwn(prefabDefinition, 'params'))
        {
            prefabDefinition['params'] = {};
        }

        return prefabDefinition['params'];
    }

    getComponentParameters(entityName, componentType) {

        let entityDefinition = this._getEntityDefinition(entityName);

        if (!Object.hasOwn(entityDefinition, 'components'))
        {
            PsyanimDebug.error('No components found for entity named: ' + entityName);
            return;
        }

        let componentDefinitions = entityDefinition['components'];

        let componentDefinition = componentDefinitions
            .find(c => c.type == componentType);

        if (!componentDefinition)
        {
            PsyanimDebug.error('Failed to find component type: ' + componentType.name);
            return;
        }

        if (!Object.hasOwn(componentDefinition, 'params'))
        {
            componentDefinition['params'] = {};
        }

        return componentDefinition['params'];
    }

    saveSceneParameter(parameterName) {

        let parameterValue = this.getSceneParameter(parameterName);

        let trialParameter = new PsyanimJsPsychTrialParameter(
            PsyanimJsPsychTrialParameter.Type.SCENE_PARAMETER,
            parameterName, parameterValue
        );

        if (!this._trialParametersToSave.find(p => p.is(trialParameter)))
        {
            this._trialParametersToSave.push(trialParameter);
        }
    }

    savePrefabParameter(entityName, parameterName) {

        let prefabParameters = this.getPrefabParameters(entityName);
        
        let parameterValue = prefabParameters[parameterName];

        let trialParameter = new PsyanimJsPsychTrialParameter(
            PsyanimJsPsychTrialParameter.Type.PREFAB_PARAMETER,
            parameterName, parameterValue, entityName
        );

        if (!this._trialParametersToSave.find(p => p.is(trialParameter)))
        {
            this._trialParametersToSave.push(trialParameter);
        }
    }

    recordAnimationParameter(entityName, componentType, key, name) {

        this._animationParameters.push({
            entityName: entityName,

            // had to wrap this value in an object or jsPsych would try to mess with it
            componentInfo: { type: componentType }, 

            key: key,
            name: name
        });
    }

    saveComponentParameter(entityName, componentType, parameterName) {

        let componentParameters = this.getComponentParameters(entityName, componentType);

        let parameterValue = componentParameters[parameterName];

        let trialParameter = new PsyanimJsPsychTrialParameter(
            PsyanimJsPsychTrialParameter.Type.COMPONENT_PARAMETER,
            parameterName, parameterValue, entityName, componentType
        );

        if (!this._trialParametersToSave.find(p => p.is(trialParameter)))
        {
            this._trialParametersToSave.push(trialParameter);
        }
    }
}